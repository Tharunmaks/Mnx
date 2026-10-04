// Writes the local model's exact system prompt and tool list to
// training/system_prompt.json so the dataset generator uses the same text
// Mnx sends at runtime. Run: node training/export_prompt.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { STABLE_PROMPT, LOCAL_TOOLS } from "../local.js";
import { MORE_NAMES } from "../tools-more.js";

const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "system_prompt.json");
fs.writeFileSync(out, JSON.stringify({ stable_prompt: STABLE_PROMPT, tools: [...LOCAL_TOOLS.map((t) => t.function.name), ...MORE_NAMES] }, null, 2) + "\n");
console.log(`Wrote ${out}`);
