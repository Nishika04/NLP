const http = require('http');
const httpProxy = require('http-proxy');

const proxy = httpProxy.createProxyServer({});

const server = http.createServer((req, res) => {
    // Set the CORS headers here
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Forward the request to the target server
    proxy.web(req, res, { target: 'http://127.0.0.1:5000/submit_data' });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`Proxy server is running on port ${PORT}`);
});
