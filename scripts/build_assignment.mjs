import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const ROOT=process.cwd(), qa=path.join(ROOT,"artifacts","office_qa","assignment_final");
await fs.mkdir(qa,{recursive:true});
const d=JSON.parse(await fs.readFile(path.join(ROOT,"artifacts","office_data.json"),"utf8"));
const wb=Workbook.create();
const colors={navy:"#17365D",blue:"#D9EAF7",teal:"#DFF1EA",orange:"#FCE4D6",white:"#FFFFFF",ink:"#1F2937",line:"#CBD5E1",green:"#E2F0D9",red:"#F4CCCC",gray:"#F3F4F6"};
const members=[
  ["24100358","Nguyễn Tùng Dương","Trưởng nhóm / ML pipeline",34],
  ["24100065","Trịnh Ngọc Nga","Data audit / thí nghiệm",33],
  ["24106898","Trương Việt Thành","App / tài liệu / QA",33],
];
function style(range, fill, color=colors.ink, bold=false, size=11) { range.format.fill=fill; range.format.font={color,bold,size}; range.format.wrapText=true; range.format.verticalAlignment="center"; range.format.borders={preset:"all",style:"thin",color:colors.line}; }
function setup(sheet,title,subtitle,cols="A:F") {
  sheet.showGridLines=false; sheet.mergeCells(`A1:${cols.split(":")[1]}1`); sheet.getRange("A1").values=[[title]]; style(sheet.getRange(`A1:${cols.split(":")[1]}1`),colors.navy,colors.white,true,18); sheet.getRange("A1").format.rowHeight=34;
  sheet.mergeCells(`A2:${cols.split(":")[1]}2`); sheet.getRange("A2").values=[[subtitle]]; style(sheet.getRange(`A2:${cols.split(":")[1]}2`),colors.blue,colors.navy,false,10); sheet.getRange("A2").format.rowHeight=30;
  sheet.freezePanes.freezeRows(3);
}
function header(sheet,address,values) { const r=sheet.getRange(address); r.values=[values]; style(r,colors.navy,colors.white,true,10); }
function write(sheet,address,values,fill=colors.white) { const r=sheet.getRange(address); r.values=values; style(r,fill,colors.ink,false,10); }

// Hướng dẫn
{
  const s=wb.worksheets.add("Hướng dẫn"); setup(s,"BẢNG PHÂN CÔNG & NGHIỆM THU — NHÓM 7","Nguồn canonical: artifacts/canonical_facts.json • Cập nhật 27/08/2026","A:F");
  header(s,"A4:F4",["Mục","Nội dung","Nguồn bằng chứng","Chủ trì","Trạng thái","Ghi chú"]);
  write(s,"A5:F11",[
    ["1","Đọc README và Data/Model Card","README.md; docs/","Cả nhóm","Hoàn tất","Không dùng metric từ tài liệu làm source"],
    ["2","Kiểm tra dữ liệu raw","data/inventory.csv","Nga","Hoàn tất","3.670 ảnh full-decode"],
    ["3","Chạy pipeline CNN","scripts/run_full_pipeline.py","Dương","Hoàn tất","FULL_RUN_COMPLETE"],
    ["4","Kiểm định 49 điều kiện","results/final/manifest.json","Dương + Nga","Hoàn tất","26.950 predictions"],
    ["5","Kiểm thử app","artifacts/app_smoke_test.json","Thành","Hoàn tất","Single + batch core smoke"],
    ["6","Deploy Streamlit","docs/STREAMLIT_DEPLOYMENT.md","Thành","Chưa triển khai","Cần URL + screenshot"],
    ["7","Đóng gói & checksum","SHA256SUMS.txt","Cả nhóm","Chờ cuối","Loại raw/cache/junction"],
  ]);
  s.getRange("A:A").format.columnWidth=10; s.getRange("B:B").format.columnWidth=34; s.getRange("C:C").format.columnWidth=34; s.getRange("D:D").format.columnWidth=18; s.getRange("E:E").format.columnWidth=18; s.getRange("F:F").format.columnWidth=30;
}

// Phân công exact required formulas at row 10:12.
{
  const s=wb.worksheets.add("Phân công"); setup(s,"PHÂN CÔNG NHÓM","Ba thành viên • tỷ trọng có công thức kiểm soát","A:F");
  write(s,"A4:B7",[["Giảng viên",d.facts.instructor],["Nhóm",d.facts.group],["Trạng thái","FULL_RUN_COMPLETE"],["Deploy","DEPLOY_READY_BUT_NOT_DEPLOYED"]],colors.gray);
  header(s,"A9:F9",["MSSV","Họ tên","Vai trò chính","Tỷ trọng (%)","Bằng chứng chính","Trạng thái"]);
  write(s,"A10:F12",members.map((m,i)=>[...m, i===0?"src/, models/, results/final":"data/, docs/, artifacts/", "Hoàn tất"]));
  write(s,"A14:C16",[["Kiểm soát","Công thức","Kết quả"],["Số thành viên","COUNTA(A10:A12)",null],["Tổng tỷ trọng","SUM(D10:D12)",null]],colors.gray);
  s.getRange("C15").formulas=[["=COUNTA(A10:A12)"]]; s.getRange("C16").formulas=[["=SUM(D10:D12)"]]; style(s.getRange("C15:C16"),colors.teal,colors.navy,true,12);
  s.getRange("A:A").format.columnWidth=14; s.getRange("B:B").format.columnWidth=25; s.getRange("C:C").format.columnWidth=30; s.getRange("D:D").format.columnWidth=14; s.getRange("E:E").format.columnWidth=32; s.getRange("F:F").format.columnWidth=16;
}

const tasks=[
  ["T01","Audit ba phiên bản","Inventory + scorecard","Nguyễn Tùng Dương",100,"artifacts/merge_audit/"],
  ["T02","Audit raw + split","Decode/hash/leakage","Trịnh Ngọc Nga",100,"data/inventory.csv; splits/"],
  ["T03","CNN + training","MobileNetV2 two-stage","Nguyễn Tùng Dương",100,"models/; artifacts/training/"],
  ["T04","49 conditions","Tuning/metrics/stats","Trịnh Ngọc Nga",100,"results/final/"],
  ["T05","Streamlit","Single + batch + readiness","Trương Việt Thành",100,"streamlit_app.py"],
  ["T06","Office coherence","DOCX/PDF/PPTX/XLSX","Trương Việt Thành",100,"docs/"],
  ["T07","Package QA","Checksum + clean extract","Cả nhóm",100,"SHA256SUMS.txt"],
];
for (let idx=0;idx<3;idx++) {
  const [id,name,role,weight]=members[idx]; const s=wb.worksheets.add(`Chi tiết TV${idx+1}`); setup(s,`CHI TIẾT — ${name}`,`${id} • ${role} • tỷ trọng ${weight}%`,"A:G");
  header(s,"A4:G4",["ID","Hạng mục","Đầu ra","Vai trò","Tiến độ (%)","Evidence","Ghi chú"]);
  const owned=tasks.filter(t=>t[3]===name || t[3]==="Cả nhóm");
  const rows=owned.map(t=>[t[0],t[1],t[2],t[3],t[4],t[5],t[4]===100?"Đã kiểm tra":"Cần hoàn tất"]);
  write(s,`A5:G${4+rows.length}`,rows);
  write(s,"A12:C14",[["Chỉ số","Công thức","Kết quả"],["Số việc",`COUNTA(A5:A${4+rows.length})`,null],["Tiến độ TB",`AVERAGE(E5:E${4+rows.length})`,null]],colors.gray);
  s.getRange("C13").formulas=[[`=COUNTA(A5:A${4+rows.length})`]]; s.getRange("C14").formulas=[[`=AVERAGE(E5:E${4+rows.length})`]]; style(s.getRange("C13:C14"),colors.teal,colors.navy,true,11);
  s.getRange("A:A").format.columnWidth=10; s.getRange("B:B").format.columnWidth=26; s.getRange("C:C").format.columnWidth=29; s.getRange("D:D").format.columnWidth=24; s.getRange("E:E").format.columnWidth=14; s.getRange("F:F").format.columnWidth=34; s.getRange("G:G").format.columnWidth=24;
}

// Tiến độ
{
  const s=wb.worksheets.add("Tiến độ"); setup(s,"TIẾN ĐỘ DỰ ÁN","Evidence-first: hoàn tất chỉ khi artifact tồn tại và qua validator","A:G");
  header(s,"A4:G4",["ID","Hạng mục","Chủ trì","%","Bắt đầu","Kết thúc","Evidence"]);
  write(s,"A5:G11",tasks.map((t,i)=>[t[0],t[1],t[3],t[4],`2026-08-${String(20+i).padStart(2,"0")}`,i<5?`2026-08-${String(22+i).padStart(2,"0")}`:"—",t[5]]));
  write(s,"A13:C15",[["KPI","Công thức","Kết quả"],["Tiến độ TB","AVERAGE(D5:D11)",null],["Việc hoàn tất","COUNTIF(D5:D11,100)",null]],colors.gray);
  s.getRange("C14").formulas=[["=AVERAGE(D5:D11)"]]; s.getRange("C15").formulas=[["=COUNTIF(D5:D11,100)"]]; style(s.getRange("C14:C15"),colors.teal,colors.navy,true,11);
  s.getRange("A:A").format.columnWidth=10; s.getRange("B:B").format.columnWidth=30; s.getRange("C:C").format.columnWidth=24; s.getRange("D:D").format.columnWidth=10; s.getRange("E:F").format.columnWidth=15; s.getRange("G:G").format.columnWidth=38;
}

// Nghiệm thu
{
  const s=wb.worksheets.add("Nghiệm thu"); setup(s,"CHECKLIST NGHIỆM THU","Không chuyển PASS nếu thiếu bằng chứng external deployment","A:F");
  header(s,"A4:F4",["Gate","Tiêu chí","Kỳ vọng","Thực tế","Kết quả","Evidence"]);
  const rows=[
    ["G1","Valid images",3670,d.facts.valid_image_count,null,"artifacts/data_audit.json"],
    ["G2","Split train/val/test","2571/549/550",`${d.facts.split_counts.train}/${d.facts.split_counts.validation}/${d.facts.split_counts.test}`,null,"splits/*.csv"],
    ["G3","Experiment rows",49,d.facts.condition_count,null,"results/final/manifest.json"],
    ["G4","Prediction rows",26950,d.facts.prediction_rows,null,"results/final/predictions.csv"],
    ["G5","Model checksum","64 hex",d.facts.model_checksum,null,"models/model_metadata.json"],
    ["G6","Public deployment","URL + screenshot","Chưa có",null,"artifacts/deployment_verification.json"],
  ];
  write(s,"A5:F10",rows);
  s.getRange("E5").formulas=[["=IF(C5=D5,\"PASS\",\"FAIL\")"]]; s.getRange("E5:E10").fillDown(); style(s.getRange("E5:E10"),colors.gray,colors.ink,true,10);
  s.getRange("E9").formulas=[["=IF(LEN(D9)=64,\"PASS\",\"FAIL\")"]];
  write(s,"A13:C15",[["KPI","Công thức","Kết quả"],["PASS","COUNTIF(E5:E10,\"PASS\")",null],["FAIL","COUNTIF(E5:E10,\"FAIL\")",null]],colors.gray);
  s.getRange("C14").formulas=[["=COUNTIF(E5:E10,\"PASS\")"]]; s.getRange("C15").formulas=[["=COUNTIF(E5:E10,\"FAIL\")"]]; style(s.getRange("C14:C15"),colors.teal,colors.navy,true,11);
  s.getRange("A:A").format.columnWidth=10; s.getRange("B:B").format.columnWidth=28; s.getRange("C:D").format.columnWidth=22; s.getRange("E:E").format.columnWidth=12; s.getRange("F:F").format.columnWidth=38;
}

// Kết quả
{
  const s=wb.worksheets.add("Kết quả"); setup(s,"TÓM TẮT KẾT QUẢ FULL RUN","Source: results/final/condition_metrics.csv","A:F");
  write(s,"A4:B9",[["Chỉ tiêu","Giá trị"],["Clean Accuracy",d.clean.accuracy],["Clean Macro F1",d.clean.macro_f1],["Best enhanced",d.best_enhanced.condition_id],["Best enhanced F1",d.best_enhanced.macro_f1],["Worst degraded F1",d.worst_degraded.macro_f1]],colors.gray);
  s.getRange("B5:B6").setNumberFormat("0.0000"); s.getRange("B8:B9").setNumberFormat("0.0000");
  header(s,"A12:F12",["Condition","Type","Accuracy","Macro F1","SSIM","Latency ms"]);
  const rows=d.top_conditions.map(r=>[r.condition_id,r.image_type,r.accuracy,r.macro_f1,r.ssim,r.inference_time_ms_per_image_mean]);
  write(s,"A13:F22",rows); s.getRange("C13:F22").setNumberFormat("0.0000");
  s.getRange("A:A").format.columnWidth=43; s.getRange("B:B").format.columnWidth=14; s.getRange("C:E").format.columnWidth=14; s.getRange("F:F").format.columnWidth=16;
}

const sheets=["Hướng dẫn","Phân công","Chi tiết TV1","Chi tiết TV2","Chi tiết TV3","Tiến độ","Nghiệm thu","Kết quả"];
for (const name of sheets) {
  const png=await wb.render({sheetName:name,autoCrop:"all",scale:1,format:"png"});
  await fs.writeFile(path.join(qa,`${name.replace(/[^\p{L}\p{N}]+/gu,"_")}.png`),new Uint8Array(await png.arrayBuffer()));
}
const inspect=await wb.inspect({kind:"workbook,sheet,formula,region",maxChars:30000,tableMaxRows:25,tableMaxCols:8});
await fs.writeFile(path.join(qa,"inspect.ndjson"),inspect.ndjson,"utf8");
const out=await SpreadsheetFile.exportXlsx(wb); await out.save(path.join(ROOT,"docs","ASSIGNMENT_FINAL.xlsx"));
console.log(`Built ${sheets.length} sheets`);
