// Import required modules
const express = require('express');
const http = require('http');
const https = require('https');
const url = require('url');
const axios = require('axios');

// Create Express application
const app = express();

// // Use cors middleware
// const cors = require('cors');
// app.use(cors());

// Import the dotenv package
require('dotenv').config();

// Use the environment variable for the server URL
const serverURL = process.env.SERVER_URL || 'http://localhost:8503';
const PORT = process.env.PORT || 8501; // Port on which the proxy server will listen
const MAX_ACTIVE_MODELS = parseInt(process.env.MAX_ACTIVE_MODELS || '0', 10); // 0 = disabled

// In-memory LRU tracking for models when dynamic model loading is active
const activeModels = []; // Array of model names, head is newest, tail is oldest

const markModelUsed = (modelName) => {
  const index = activeModels.indexOf(modelName);
  if (index !== -1) {
    activeModels.splice(index, 1);
  }
  activeModels.unshift(modelName);
};

const ensureModelLoaded = async (modelName) => {
  if (!MAX_ACTIVE_MODELS || MAX_ACTIVE_MODELS <= 0 || !modelName) {
    return;
  }
  try {
    // Check if model is already ready
    const readyRes = await axios.get(`${serverURL}/v2/models/${modelName}/ready`).catch(() => null);
    if (readyRes && readyRes.status === 200) {
      markModelUsed(modelName);
      return;
    }

    // Model is not loaded or not ready -> load it
    console.log(`[LRU Manager] Loading model '${modelName}'...`);
    await axios.post(`${serverURL}/v2/repository/models/${modelName}/load`);
    markModelUsed(modelName);

    // Evict oldest models if exceeding MAX_ACTIVE_MODELS
    while (activeModels.length > MAX_ACTIVE_MODELS) {
      const oldestModel = activeModels.pop();
      if (oldestModel && oldestModel !== modelName) {
        console.log(`[LRU Manager] Evicting oldest model '${oldestModel}'...`);
        await axios.post(`${serverURL}/v2/repository/models/${oldestModel}/unload`).catch((err) => {
          console.warn(`[LRU Manager] Failed to unload '${oldestModel}':`, err.message);
        });
      }
    }
  } catch (error) {
    console.warn(`[LRU Manager] Failed to ensure model '${modelName}' is loaded:`, error.message);
  }
};

const handleProxyRequest = async (req, res, targetURL) => {
  const proxyRequest = (targetURL.protocol === 'https:') ? https.request : http.request;

  const options = {
    hostname: targetURL.hostname,
    port: targetURL.port,
    path: targetURL.path,
    method: req.method,
    headers: req.headers,
  };

  const proxyReq = proxyRequest(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    console.error('Proxy request error:', err);
    if (!res.headersSent) {
      res.status(500).send('Proxy request error');
    }
    proxyReq.end();
  });

  req.pipe(proxyReq);
};

function transformKoinaSpectrum(spectrum) {
  const mzOutput = spectrum.outputs.find(output => output.name === 'mz');
  const intensityOutput = spectrum.outputs.find(output => output.name === 'intensities');

  if (!mzOutput || !intensityOutput) {
    throw new Error('mz or intensities array not found in outputs');
  }

  const mzs = mzOutput.data || [];
  const intensities = intensityOutput.data || [];

  return {
    mzs: mzs.filter((mz, index) => intensities[index] > 1e-4),
    intensities: intensities.filter((intensity) => intensity > 1e-4),
  };
}

function usiGetInterpretation(inputString) {
  const parts = inputString.split(':');
  return parts.length > 5 ? parts[5] : '';
}

async function createReqInput(modelName) {
  try {
    const response = await axios.get(`${serverURL}/v2/models/${modelName}/config`);
    modelConfig = response.data; // Send the data from the URL back to the client
  } catch (error) {
    console.error('Failed to fetch config:', error);
    res.status(500).send('Failed to fetch config');
  }

  return modelConfig['input'].map(({ name, data_type }) => ({
    name,
    datatype: data_type.replace('TYPE_', '').replace('STRING', 'BYTES'),
    shape: [1, 1]
  }));
}

const createPayload = async (req) => {
  // Extract the wildcard (*) part from the URL
  const modelName = req.params[0];
  let usiPeptideSequence;
  let usiPrecursorCharge;
  const usiInterpretation = usiGetInterpretation(req.query.usi ? req.query.usi : '');

  [usiPeptideSequence, precursorCharge] = usiInterpretation.split("/");
  precursorCharge = parseInt(usiPrecursorCharge);

  let payload = await createReqInput(modelName)
  // Define the static JSON payload
  payloadInputs = payload.map(input => {
    if (req.query[input.name] === undefined && input.name !== "peptide_sequences" && input.name !== "precursor_charges") {
      throw new Error(`Required input ${input.name} was not provided`);
    }

    let dataValue;

    if (input.name === "peptide_sequences" && req.query[input.name] === undefined) {
      dataValue = usiPeptideSequence;
    } else if (input.name === "precursor_charges" && req.query[input.name] === undefined) {
      dataValue = usiPrecursorCharge;
    } else if (input.datatype != "BYTES") {
      dataValue = parseFloat(req.query[input.name]);
    } else {
      dataValue = req.query[input.name];
    }

    return {
      ...input,
      data: [dataValue]
    }
  }
  );

  payload = {
    "id": "0",
    "inputs": payloadInputs
  };

  // Define options for the POST request
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    data: JSON.stringify(payload)
  };
  return options
}


// Route handler for /v2/models/*/usi endpoint
app.get('/v2/models/*/usi', async (req, res) => {
  const validModelNames = ['AlphaPeptDeep_ms2_generic',
    'Prosit_2019_intensity',
    'Prosit_2020_intensity_CID',
    'Prosit_2020_intensity_HCD',
    'Prosit_2020_intensity_TMT',
    'Prosit_2023_intensity_timsTOF',
    'Prosit_2024_intensity_PTMs_gl',
    'Prosit_2024_intensity_cit',
    'UniSpec',
    'ms2pip_CID_TMT',
    'ms2pip_HCD2021',
    'ms2pip_Immuno_HCD',
    'ms2pip_TTOF5600',
    'ms2pip_iTRAQphospho',
    'ms2pip_timsTOF2023',
    'ms2pip_timsTOF2024'];

  const modelName = req.params[0];

  // validate the model name against the allow list
  if (!validModelNames.includes(modelName)) {
    return res.status(400).send('[ERROR] Invalid model name');
  }

  // Make the POST request to the target URL using axios
  try {
    // Modify the requested path to /v2/models/*/infer
    const targetURL = `${serverURL}/v2/models/${modelName}/infer`;

    const options = await createPayload(req)
    const response = await axios(targetURL, options);
    // Handle the response data as needed
    let data = transformKoinaSpectrum(response.data);
    res.json({
      attributes: { "origin": "Koina", "model": modelName, "payload": options['data'] },
      ...data
    });
  } catch (error) {
    // Use different status codes based on the type of error
    if (error.message === 'mz or intensities array not found in outputs') {
      res.status(400).send(`[ERROR] ${error.message}`);
    } else {
      res.status(500).send(`[ERROR] ${error.message}`);
    }
  }
});

// Middleware for all endpoints
app.use(async (req, res, next) => {
  // Extract model name if the request targets a model endpoint (e.g. /v2/models/<modelName>/...)
  const modelMatch = req.url.match(/^\/v2\/models\/([^/]+)/);
  if (modelMatch && modelMatch[1]) {
    await ensureModelLoaded(modelMatch[1]);
  }

  // Parse the request URL
  const targetURL = url.parse(serverURL + req.url);
  await handleProxyRequest(req, res, targetURL);
});

// Start the proxy server
app.listen(PORT, () => {
  console.log(`Proxy server listening on port ${PORT} forwarding requests to ${serverURL}`);
});
