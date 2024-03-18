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

const handleProxyRequest = (req, res, targetURL) => {
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


// Route handler for /v2/models/*/usi endpoint
app.get('/v2/models/*/usi', async (req, res) => {
  // Extract the wildcard (*) part from the URL
  const modelName = req.params[0];
  let peptideSequence;
  let precursorCharge;
  const usiInterpretation = usiGetInterpretation(req.query.usi ? req.query.usi : '');

  [peptideSequence, precursorCharge] = usiInterpretation.split("/");
  precursorCharge = parseInt(precursorCharge);

  let payload = await createReqInput(modelName)
  // Define the static JSON payload
  payloadInputs = payload.map(input => {
    return {
      ...input,
      data: [
        input.name === "peptide_sequences" && req.query[input.name] === undefined
          ? peptideSequence
          : input.name === "precursor_charges" && req.query[input.name] === undefined
            ? precursorCharge
            : input.datatype != "BYTES"
              ? parseFloat(req.query[input.name])
              : req.query[input.name]]
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

// Modify the requested path to /v2/models/*/infer
const targetURL = `${serverURL}/v2/models/${modelName}/infer`;

// Make the POST request to the target URL using axios
try {
  const response = await axios(targetURL, options);
  // Handle the response data as needed
  let data = transformKoinaSpectrum(response.data);
  res.json({
    attributes: { "origin": "Koina", "model": modelName, "payload": payload },
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
  // Parse the request URL
  const targetURL = url.parse(serverURL + req.url);
  await handleProxyRequest(req, res, targetURL);
});

// Start the proxy server
app.listen(PORT, () => {
  console.log(`Proxy server listening on port ${PORT} forwarding requests to ${serverURL}`);
});


