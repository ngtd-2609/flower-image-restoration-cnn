import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = fileURLToPath(new URL("../docs/ASSIGNMENT_FINAL.xlsx", import.meta.url));
const outputDir = fileURLToPath(new URL("../artifacts/office_qa/assignment_before/", import.meta.url));
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const summary = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 30,
  tableMaxCols: 12,
  options: { maxResults: 200 },
});
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(`${outputDir}/inspect.ndjson`, summary.ndjson, "utf8");
for (const sheet of workbook.worksheets.items) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1.5, format: "png" });
  await fs.writeFile(`${outputDir}/${sheet.name.replace(/[^a-z0-9_-]/gi, "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}
console.log(summary.ndjson);
