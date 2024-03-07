// Import required modules
const express = require('express');
const http = require('http');
const https = require('https');
const url = require('url');

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

// Route handler for /v2/models/*/use endpoint
app.get('/v2/models/*/use', (req, res) => {
  // Extract the wildcard (*) part from the URL
  const modelName = req.params[0];
  const peptideSequence = req.query.peptide_sequence;

  // Define the static JSON payload
  const payload = {
    "id": "0",
    "inputs": [
      {
        "name": "peptide_sequences",
        "shape": [1,1],
        "datatype": "BYTES",
        "data": [ peptideSequence ]
      }
    ]
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
      res.json(data);
    })
    .catch(error => {
      console.error('There was a problem with your fetch operation:', error);
      res.status(500).send('Internal Server Error');
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
