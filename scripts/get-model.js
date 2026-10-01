// Downloads a GGUF model from Google Drive into models/.
// Usage: npm run get-model -- [driveFileIdOrUrl] [outputName]
// The Drive file must be shared as "Anyone with the link"; otherwise download
// it in your browser and move it into the models/ folder.
import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const DEFAULT_ID = "1I5IUAq5ckpC1cPHfBlU766VC1n66Rdlv";
const arg = process.argv[2] || DEFAULT_ID;
const id = arg.match(/\/d\/([\w-]+)/)?.[1] || arg.match(/[?&]id=([\w-]+)/)?.[1] || arg;
const name = process.argv[3] || "llama-3.2-1b-instruct.Q4_K_M.gguf";
const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "models");
const out = path.join(dir, name);

fs.mkdirSync(dir, { recursive: true });
const url = `https://drive.usercontent.google.com/download?id=${encodeURIComponent(id)}&export=download&confirm=t`;
console.log(`Downloading ${id} → ${path.relative(process.cwd(), out)}`);

const res = await fetch(url, { redirect: "follow" });
const type = res.headers.get("content-type") || "";
if (!res.ok || type.includes("text/html")) {
  console.error(
    `\nGoogle Drive didn't return the file (HTTP ${res.status}). Make sure it's shared as "Anyone with the link",\n` +
      `or download it in your browser and move it to: ${out}`,
  );
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
    process.stdout.write(`\r  ${(done / 1048576).toFixed(0)} MB${total ? ` / ${(total / 1048576).toFixed(0)} MB` : ""}`);
  }
});
await pipeline(body, fs.createWriteStream(`${out}.part`));
fs.renameSync(`${out}.part`, out);
console.log(`\nSaved ${out}. Start Mnx with: npm start`);
