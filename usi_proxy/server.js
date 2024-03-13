// Import required modules
const express = require('express');
const http = require('http');
const https = require('https');
const url = require('url');
const axios = require('axios');

// Create Express application
const app = express();
const PORT = 3000; // Port on which the proxy server will listen

// Function to handle proxy requests
const handleProxyRequest = (req, res, targetURL) => {
  // Determine whether to use HTTP or HTTPS
  const proxyRequest = (targetURL.protocol === 'https:') ? https.request : http.request;
  const proxyReq = proxyRequest(targetURL, (proxyRes) => {
    console.log('Response status code:', proxyRes.statusCode);

    // Log data received from the target server
    let responseData = '';
    proxyRes.on('data', (chunk) => {
      responseData += chunk;
    });

    proxyRes.on('end', () => {
      console.log('Response data:', responseData);
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      res.end(responseData); // End the client response
    });
  });

  // Handle errors
  proxyReq.on('error', (err) => {
    console.error('Proxy error:', err);
    res.status(500).send('Proxy error');
  });

  // Pipe the incoming request body to the proxy request (if any)
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
    const response = await axios.get(`http://serving:8501/v2/models/${modelName}/config`);
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


// Route handler for /v2/models/*/use endpoint
app.get('/v2/models/*/use', async (req, res) => {
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
    body: JSON.stringify(payload)
  };

  // Modify the requested path to /v2/models/*/infer
  const targetURL = `http://serving:8501/v2/models/${modelName}/infer`;

  // Make the POST request to the target URL
  fetch(targetURL, options)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      // Handle the response data as needed
      data = transformKoinaSpectrum(data);
      res.json({
        attributes: { "origin": "Koina", "model": modelName, "payload": payload },
        ...data
      });
    })
    .catch(error => {
      console.error('There was a problem with your fetch operation:', error);
      if (error.message === 'mz or intensities array not found in outputs') {
        res.status(400).send(`[ERROR] ${error.message}`);
      } else {
        res.status(500).send('Internal Server Error');
      }
    });
});



// Middleware for all endpoints
app.use((req, res, next) => {
  // Parse the request URL
  const targetURL = url.parse('http://serving:8501' + req.url);
  handleProxyRequest(req, res, targetURL);
});

// Start the proxy server
app.listen(PORT, () => {
  console.log(`Proxy server listening on port ${PORT}`);
});
