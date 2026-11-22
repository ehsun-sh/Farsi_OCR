// Tiny static server for local testing: node scripts/serve.mjs [port]
import {createServer} from 'node:http';
import {readFile, stat} from 'node:fs/promises';
import {extname, join, normalize} from 'node:path';
import {fileURLToPath} from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.argv[2] || process.env.PORT || 8788);
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.bcmap': 'application/octet-stream', '.pfb': 'application/octet-stream', '.ttf': 'font/ttf', '.png': 'image/png',
};

createServer(async (req, res) => {
  try {
    let path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([.][.][/\\])+/, '');
    let file = join(dist, path);
    if ((await stat(file).catch(() => null))?.isDirectory()) file = join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, {'content-type': types[extname(file)] || 'application/octet-stream'});
    res.end(body);
  } catch {
    res.writeHead(404, {'content-type': 'text/plain'});
    res.end('Not found');
  }
}).listen(port, '127.0.0.1', () => console.log(`Farsi OCR: http://127.0.0.1:${port}/`));
