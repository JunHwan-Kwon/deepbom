import { buildZenodoArchive } from "./zenodo-release-lib.mjs";

const args = new Set(process.argv.slice(2));
const kindArg = process.argv.find((value) => value.startsWith("--kind="));
const outArg = process.argv.find((value) => value.startsWith("--out-dir="));
const validationArg = process.argv.find((value) => value.startsWith("--validation-root="));

const summary = buildZenodoArchive({
  kind: kindArg?.slice("--kind=".length) || "software",
  allowDirty: args.has("--allow-dirty"),
  outputDirectory: outArg?.slice("--out-dir=".length) || ".local-validation/zenodo-citation",
  validationRoot: validationArg?.slice("--validation-root=".length) || ".local-validation/supported-formats/latest",
});

console.log(JSON.stringify(summary, null, 2));
