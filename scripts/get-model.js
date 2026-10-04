// Downloads a GGUF model into models/.
//
//   npm run get-model                      # your Mnx model from Hugging Face
//   npm run get-model -- <owner/repo/file.gguf>
//   npm run get-model -- <google-drive-link> [name.gguf]
//
// Private Hugging Face repos need a token: HF_TOKEN=hf_... npm run get-model
// (create one at https://huggingface.co/settings/tokens, "Read" access).
// Google Drive files must be shared as "Anyone with the link".
import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const DEFAULT = "tharunmakes/mnx-qwen2.5-3b-gguf/mnx-q4_k_m.gguf";
const arg = process.argv[2] || DEFAULT;
const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "models");

let url;
let name;
const headers = {};
const drive = arg.match(/\/d\/([\w-]+)/)?.[1] || arg.match(/[?&]id=([\w-]+)/)?.[1];
if (drive || /drive\.google\.com/.test(arg)) {
  url = `https://drive.usercontent.google.com/download?id=${encodeURIComponent(drive)}&export=download&confirm=t`;
  name = process.argv[3] || "mnx-q4_k_m.gguf";
} else {
  // owner/repo/path/to/file.gguf  (or a full huggingface.co/.../resolve/... URL)
  const m = arg.replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/(resolve|blob)\/[^/]+\//, "/").match(/^([^/]+\/[^/]+)\/(.+\.gguf)$/i);
  if (!m) {
    console.error(`Don't know how to download "${arg}". Use owner/repo/file.gguf or a Google Drive link.`);
    process.exit(1);
  }
  url = `https://huggingface.co/${m[1]}/resolve/main/${m[2].split("/").map(encodeURIComponent).join("/")}`;
  name = process.argv[3] || path.basename(m[2]);
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  if (token) headers.authorization = `Bearer ${token}`;
}

const out = path.join(dir, name);
fs.mkdirSync(dir, { recursive: true });
console.log(`Downloading → ${path.relative(process.cwd(), out)}`);

const res = await fetch(url, { headers, redirect: "follow" });
const type = res.headers.get("content-type") || "";
if (!res.ok || type.includes("text/html")) {
  const hint =
    res.status === 401 || res.status === 403 || (res.status === 404 && !headers.authorization)
      ? "This repo is private: set HF_TOKEN to a Hugging Face token with read access."
      : drive
        ? 'Make sure the Drive file is shared as "Anyone with the link", or download it in your browser.'
        : "Check the repo and file name.";
  console.error(`\nDownload failed (HTTP ${res.status}). ${hint}\nYou can also copy the .gguf file into: ${dir}`);
  process.exit(1);
}
const total = Number(res.headers.get("content-length")) || 0;
let done = 0;
let last = 0;
const body = Readable.fromWeb(res.body);
body.on("data", (c) => {
  done += c.length;
  if (Date.now() - last > 500) {
    last = Date.now();
    const pct = total ? ` (${Math.floor((done / total) * 100)}%)` : "";
    process.stdout.write(`\r  ${(done / 1048576).toFixed(0)} MB${total ? ` / ${(total / 1048576).toFixed(0)} MB` : ""}${pct}   `);
  }
});
await pipeline(body, fs.createWriteStream(`${out}.part`));
fs.renameSync(`${out}.part`, out);
console.log(`\nSaved ${out}. Start Mnx with: npm start`);
