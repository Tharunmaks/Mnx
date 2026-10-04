// node --import ./test/fixtures/use-mock-net-tools.mjs server.js
// Routes Mnx's `./tools.js` imports to mock-net-tools.mjs (test/demo only).
import { register } from "node:module";
register("data:text/javascript," + encodeURIComponent(`
  export async function resolve(spec, ctx, next) {
    if (spec === "./tools.js" && /\\/(local|server)\\.js$/.test(ctx.parentURL || ""))
      return { url: new URL("./test/fixtures/mock-net-tools.mjs", ctx.parentURL).href, shortCircuit: true };
    return next(spec, ctx);
  }`));
