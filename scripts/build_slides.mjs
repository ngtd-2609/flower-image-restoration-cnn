import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const outDir = path.join(ROOT, "artifacts", "office_qa", "slides_final");
await fs.mkdir(outDir, { recursive: true });
const d = JSON.parse(await fs.readFile(path.join(ROOT, "artifacts", "office_data.json"), "utf8"));
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { navy: "#0B1F3A", blue: "#1D4ED8", teal: "#0F766E", mint: "#DFF4EF", orange: "#EA580C", sand: "#FFF7ED", ink: "#172033", muted: "#52606D", pale: "#F5F8FC", white: "#FFFFFF", line: "#D7E0EA", red: "#B91C1C" };

function box(slide, x, y, w, h, fill=C.white, radius="rounded-xl", line=C.line) {
  return slide.shapes.add({ geometry:"roundRect", position:{left:x,top:y,width:w,height:h}, fill, line:{style:"solid",fill:line,width:1}, borderRadius:radius });
}
function text(slide, value, x, y, w, h, size=24, color=C.ink, bold=false, align="left") {
  const s=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=value; s.text.style={fontSize:size,color,bold,alignment:align,fontFamily:"Aptos"}; return s;
}
function title(slide, eyebrow, heading, sub="") {
  text(slide, eyebrow.toUpperCase(), 64, 38, 620, 24, 12, C.teal, true);
  text(slide, heading, 64, 72, 1152, 62, 34, C.navy, true);
  if(sub) text(slide, sub, 64, 137, 1120, 38, 17, C.muted);
  slide.shapes.add({geometry:"rect",position:{left:64,top:187,width:72,height:5},fill:C.orange,line:{style:"solid",fill:C.orange,width:0}});
}
function foot(slide, n) { text(slide, `NHÓM 7  •  ${String(n).padStart(2,"0")}`, 64, 680, 300, 20, 11, C.muted, true); }
function notes(slide, sources) { slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map(x=>`- ${x}`).join("\n")}\n[/Sources]`); }
async function image(slide, rel, alt, pos, fit="contain") {
  const p=path.join(ROOT, ...rel.split("/")); const bytes=await fs.readFile(p);
  slide.images.add({blob:bytes,contentType:"image/png",alt,fit,position:pos,geometry:"roundRect",borderRadius:"rounded-xl"});
}
function metricCard(slide, x, y, label, value, note, fill=C.pale) {
  box(slide,x,y,260,150,fill); text(slide,label.toUpperCase(),x+22,y+20,216,22,12,C.muted,true); text(slide,value,x+22,y+48,216,48,30,C.navy,true); text(slide,note,x+22,y+104,216,28,13,C.muted);
}
function bullets(slide, items, x, y, w, size=20, gap=58) { items.forEach((v,i)=>{ text(slide,"●",x,y+i*gap,20,28,12,C.orange,true); text(slide,v,x+28,y+i*gap,w-28,44,size,C.ink); }); }

// 1 — cover
{
  const s=deck.slides.add(); s.background.fill=C.navy;
  text(s,"BÀI TẬP LỚN XỬ LÝ ẢNH",70,64,560,28,14,"#7DD3FC",true);
  text(s,"Nhận diện 5 loại hoa\ntrong điều kiện ảnh suy giảm",70,132,760,160,46,C.white,true);
  text(s,"Một CNN cố định • 49 điều kiện • 26.950 dự đoán",72,316,700,42,21,"#D7E9FF");
  box(s,820,96,360,470,"#102B50","rounded-2xl","#254B75");
  text(s,"NHÓM 7",855,132,290,36,22,"#7DD3FC",true);
  text(s,"24100358\nNguyễn Tùng Dương\n\n24100065\nTrịnh Ngọc Nga\n\n24106898\nTrương Việt Thành",855,190,290,245,20,C.white,true);
  text(s,"GV: ThS. Nguyễn Văn Sơn\n27 • 08 • 2026",855,488,290,58,16,"#D7E9FF");
  text(s,"FULL_RUN_COMPLETE",72,628,360,25,13,"#A7F3D0",true);
  notes(s,["configs/submission_metadata.json","artifacts/full_run_metadata.json"]);
}

// 2
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"01 • Bài toán","Một câu hỏi, ba lớp bằng chứng","Suy giảm ảnh → xử lý ảnh → hiệu năng nhận diện");
  const cards=[
    ["1","Suy giảm","5 cơ chế × 3 mức","#E0E7FF"], ["2","Enhancement","33 cấu hình khóa","#DFF4EF"], ["3","CNN","MobileNetV2 duy nhất","#FFF0E6"]
  ];
  cards.forEach((c,i)=>{ const x=65+i*390; box(s,x,235,350,300,c[3]); text(s,c[0],x+24,258,52,52,36,C.navy,true); text(s,c[1],x+24,330,300,42,28,C.navy,true); text(s,c[2],x+24,386,300,58,19,C.muted); });
  text(s,"Metric chính: Macro F1  •  Metric phụ: Accuracy, PSNR, SSIM, ΔE2000, latency",66,590,1110,36,18,C.ink,true);
  foot(s,2); notes(s,["README.md","configs/degradation_matrix.json"]);
}

// 3
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"02 • Best-of-Three","Hợp nhất có provenance, không trộn mù");
  const rows=[
    ["A","44/100","Taxonomy lỗi + khung trình bày","Loại surrogate & metric"],
    ["B","73/100","Backbone kỹ thuật, tests, app","Chọn"],
    ["C","65/100","Bố cục báo cáo/slide/phân công","Chọn cấu trúc"],
  ];
  rows.forEach((r,i)=>{const y=235+i*120; box(s,65,y,1150,92,i===1?C.mint:C.pale); text(s,r[0],88,y+20,60,44,30,C.navy,true); text(s,r[1],168,y+25,125,36,22,i===1?C.teal:C.muted,true); text(s,r[2],322,y+19,510,54,19,C.ink,true); text(s,r[3],865,y+19,320,54,17,i===1?C.teal:C.red,true);});
  foot(s,3); notes(s,["artifacts/merge_audit/VERSION_SCORING.md","artifacts/merge_audit/MERGE_DECISION_MATRIX.csv"]);
}

// 4
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"03 • Dữ liệu","3.670 ảnh đã full-decode và băm SHA-256","5 lớp • 3 nhóm trùng • 1 nhóm trùng khác nhãn");
  await image(s,"figures/eda/class_distribution.png","Biểu đồ phân bố năm lớp",{left:66,top:225,width:650,height:380});
  metricCard(s,770,225,"Train","2.571","70,05%",C.white); metricCard(s,770,405,"Validation","549","14,96%",C.white);
  metricCard(s,960,405,"Test","550","14,99%",C.white);
  text(s,"0 ảnh lỗi giải mã",770,590,370,28,17,C.teal,true);
  foot(s,4); notes(s,["data/inventory.csv","artifacts/data_audit.json","splits/train.csv","splits/validation.csv","splits/test.csv"]);
}

// 5
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"04 • Leakage control","Split trước mọi biến đổi","Group = SHA-256 • Stratify = class • Seed = 42");
  const xs=[85,455,825], labels=[["TRAIN","2.571"],["VALIDATION","549"],["TEST","550"]];
  labels.forEach((r,i)=>{box(s,xs[i],260,300,190,["#E0E7FF",C.mint,C.sand][i]); text(s,r[0],xs[i]+28,290,244,28,16,C.muted,true,"center"); text(s,r[1],xs[i]+28,338,244,54,40,C.navy,true,"center");});
  text(s,"∩ PATH = 0     •     ∩ SHA-256 = 0     •     bỏ sót = 0",180,520,920,45,26,C.teal,true,"center");
  foot(s,5); notes(s,["splits/*.csv","scripts/validate_project.py"]);
}

// 6
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"05 • CNN","MobileNetV2: một model cho mọi điều kiện","ImageNet → GAP → Dropout 0,3 → Softmax(5)");
  const nodes=[["RGB + EXIF","224 × 224"],["MobileNetV2","ImageNet"],["GAP + Dropout","0,3"],["Dense Softmax","5 lớp"]];
  nodes.forEach((n,i)=>{const x=55+i*305; box(s,x,270,255,180,i===1?C.mint:C.white); text(s,n[0],x+20,310,215,40,23,C.navy,true,"center"); text(s,n[1],x+20,365,215,30,16,C.muted,true,"center"); if(i<3) text(s,"→",x+264,330,34,38,30,C.orange,true,"center");});
  text(s,`SHA-256 model: ${d.facts.model_checksum.slice(0,16)}…`,65,560,800,30,17,C.muted);
  foot(s,6); notes(s,["src/model.py","src/preprocessing.py","models/model_metadata.json","https://doi.org/10.1109/CVPR.2018.00474"]);
}

// 7
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"06 • Huấn luyện","Hai giai đoạn, checkpoint theo Validation",`${d.history_epochs} epoch thực tế • lịch sử có LR và thời gian từng epoch`);
  await image(s,"artifacts/training/learning_curves.png","Đường học train và validation",{left:90,top:215,width:760,height:410});
  box(s,895,235,300,320,C.pale); bullets(s,["Head frozen","Fine-tune tầng cuối","Early stopping","Checkpoint .keras"],925,270,240,18,62);
  foot(s,7); notes(s,["artifacts/training/history.csv","artifacts/training/learning_curves.png","src/train.py"]);
}

// 8
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"07 • Thí nghiệm","49 điều kiện, cùng 550 ảnh Test");
  metricCard(s,70,235,"Clean","1","đường chuẩn",C.white); metricCard(s,355,235,"Degraded","15","5 × 3",C.sand); metricCard(s,640,235,"Enhanced","33","config khóa",C.mint); metricCard(s,925,235,"Predictions","26.950","49 × 550","#E0E7FF");
  text(s,"Validation chọn tham số  →  khóa JSON + checksum  →  Test đúng một lần",120,485,1040,48,25,C.navy,true,"center");
  foot(s,8); notes(s,["results/final/condition_metrics.csv","configs/locked_enhancement_params.json","results/final/manifest.json"]);
}

// 9
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"08 • Tuning","Không cho Test tham gia lựa chọn","Macro F1 ↓  •  SSIM ↓  •  latency ↑");
  const steps=[["1","Toàn bộ 549 Validation"],["2","Grid theo từng cơ chế"],["3","Tie-break định trước"],["4","Khóa model + split hash"]];
  steps.forEach((r,i)=>{const x=62+i*300; box(s,x,250,270,230,i===3?C.mint:C.pale); text(s,r[0],x+24,275,48,48,34,C.orange,true); text(s,r[1],x+24,345,220,84,21,C.navy,true);});
  text(s,"Test leakage check: PASS",65,560,450,34,20,C.teal,true); foot(s,9); notes(s,["src/tuning.py","results/validation_tuning_all.csv","configs/locked_enhancement_params.json"]);
}

// 10
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"09 • Kết quả","Ảnh sạch tạo đường chuẩn mạnh");
  metricCard(s,70,225,"Clean Accuracy",`${(100*d.clean.accuracy).toFixed(2)}%`,"n = 550",C.white);
  metricCard(s,355,225,"Clean Macro F1",d.clean.macro_f1.toFixed(4),"metric chính",C.mint);
  metricCard(s,640,225,"Best enhanced",d.best_enhanced.macro_f1.toFixed(4),d.best_enhanced.condition_id,C.white);
  metricCard(s,925,225,"Worst degraded",d.worst_degraded.macro_f1.toFixed(4),d.worst_degraded.condition_id,C.sand);
  text(s,"Enhancement không mặc nhiên cải thiện nhận diện.",110,515,1060,46,28,C.navy,true,"center");
  foot(s,10); notes(s,["results/final/condition_metrics.csv"]);
}

// 11 chart
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"10 • Độ bền","Mỗi cơ chế gây một kiểu mất thông tin");
  const cats=d.degradation_summary.map(x=>x.degradation); const vals=d.degradation_summary.map(x=>x.macro_f1);
  s.charts.add("bar",{position:{left:80,top:220,width:760,height:390},categories:cats,series:[{name:"Macro F1 degraded",values:vals,fill:"accent1"}],hasLegend:false,dataLabels:{showValue:true,position:"outEnd"},xAxis:{majorGridlines:{style:"solid",fill:"#D7E0EA",width:1}}});
  box(s,885,235,300,320,C.pale); bullets(s,["So với clean","Tách theo cơ chế","Cùng Test split","Không cherry-pick"],915,270,240,18,62);
  foot(s,11); notes(s,["results/final/condition_metrics.csv"]);
}

// 12
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"11 • Ảnh ≠ Nhãn","SSIM cao không bảo đảm Macro F1 cao","Metric chất lượng ảnh là bổ sung, không thay metric nhận diện");
  const types=d.type_summary; s.charts.add("column",{position:{left:70,top:225,width:720,height:370},categories:types.map(x=>x.image_type),series:[{name:"Macro F1",values:types.map(x=>x.macro_f1),fill:"accent1"},{name:"SSIM",values:types.map(x=>x.ssim),fill:"accent2"}],hasLegend:true,dataLabels:{showValue:true,position:"outEnd"}});
  box(s,840,240,350,300,C.white); bullets(s,["PSNR/SSIM: độ trung thành","ΔE2000: sai khác màu","Macro F1: nhận diện","Latency: chi phí"],870,270,290,18,58);
  foot(s,12); notes(s,["results/final/condition_metrics.csv","src/image_metrics.py","https://doi.org/10.1109/TIP.2003.819861"]);
}

// 13
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"12 • Thống kê","So sánh cặp trên cùng ảnh","Bootstrap CI 95% + McNemar + Holm");
  const stages=[["Predictions","cùng 550 ảnh"],["Difference","Accuracy + F1"],["Bootstrap","2.000 mẫu"],["McNemar","đúng ↔ sai"],["Holm","33 so sánh"]];
  stages.forEach((r,i)=>{const x=52+i*242; box(s,x,260,215,185,i===4?C.mint:C.pale); text(s,r[0],x+18,292,179,34,20,C.navy,true,"center"); text(s,r[1],x+18,350,179,38,16,C.muted,true,"center");});
  text(s,`${d.significant_rows} / 66 hàng metric bác bỏ H0 sau Holm`,190,520,900,44,25,C.teal,true,"center");
  foot(s,13); notes(s,["results/final/statistical_tests.csv","src/classification_metrics.py","https://doi.org/10.1007/BF02295996"]);
}

// 14
{
  const s=deck.slides.add(); s.background.fill=C.pale; title(s,"13 • Phân tích lỗi","Từ aggregate về từng ảnh");
  const groups=[["Always wrong","Khó cố hữu"],["Harmed by degradation","Mất dấu hiệu"],["Recovered","Enhancement cứu"],["Harmed by enhancement","Can thiệp quá mức"]];
  groups.forEach((r,i)=>{const x=62+(i%2)*580,y=235+Math.floor(i/2)*170; box(s,x,y,530,135,i===2?C.mint:C.white); text(s,r[0],x+24,y+24,475,30,22,C.navy,true); text(s,r[1],x+24,y+67,475,28,17,C.muted);});
  text(s,"Trace: condition_id • path • SHA-256 • true/pred • confidence Δ",115,585,1050,32,18,C.ink,true,"center");
  foot(s,14); notes(s,["results/final/error_analysis.csv","results/final/top_confusion_pairs.csv"]);
}

// 15
{
  const s=deck.slides.add(); s.background.fill=C.white; title(s,"14 • Bàn giao","App local sẵn sàng; deployment công khai chưa xác minh");
  const left=[["✓","Model .keras + checksum"],["✓","Notebook Run-All evidence"],["✓","Tests + validators"],["✓","Word • PDF • PPTX • XLSX"]];
  left.forEach((r,i)=>{text(s,r[0],80,230+i*70,30,30,22,C.teal,true); text(s,r[1],125,228+i*70,500,40,20,C.ink,true);});
  box(s,720,225,450,300,C.sand); text(s,"TRẠNG THÁI TRUNG THỰC",755,260,380,28,14,C.orange,true); text(s,"DEPLOY_READY\nBUT NOT DEPLOYED",755,310,380,95,31,C.navy,true); text(s,"Cần URL + ảnh chụp xác minh\ntrong môi trường có Internet.",755,430,360,55,17,C.muted);
  foot(s,15); notes(s,["streamlit_app.py","docs/STREAMLIT_DEPLOYMENT.md","artifacts/canonical_facts.json"]);
}

// 16
{
  const s=deck.slides.add(); s.background.fill=C.navy;
  text(s,"KẾT LUẬN",70,70,300,28,14,"#7DD3FC",true);
  text(s,"Xử lý ảnh phải được\nđánh giá bằng mục tiêu cuối.",70,145,780,120,42,C.white,true);
  bullets(s,["Một CNN thật, không surrogate","49 điều kiện có kiểm soát","Validation chọn — Test xác nhận","Coherence xuyên artifact"],80,330,650,20,58);
  box(s,840,165,330,325,"#102B50","rounded-2xl","#254B75"); text(s,"Clean Macro F1",875,205,260,24,14,"#7DD3FC",true); text(s,d.clean.macro_f1.toFixed(4),875,245,260,58,42,C.white,true); text(s,"Best enhanced",875,335,260,24,14,"#7DD3FC",true); text(s,d.best_enhanced.macro_f1.toFixed(4),875,375,260,58,42,C.white,true);
  text(s,"Cảm ơn — Q&A",70,635,350,30,20,"#A7F3D0",true); notes(s,["artifacts/canonical_facts.json","results/final/condition_metrics.csv"]);
}

for (let i=0;i<deck.slides.items.length;i++) {
  const slide=deck.slides.items[i], stem=`slide-${String(i+1).padStart(2,"0")}`;
  const png=await deck.export({slide,format:"png",scale:1}); await fs.writeFile(path.join(outDir,`${stem}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await slide.export({format:"layout"}); await fs.writeFile(path.join(outDir,`${stem}.layout.json`),await layout.text());
}
const montage=await deck.export({format:"webp",montage:true,scale:1}); await fs.writeFile(path.join(outDir,"deck-montage.webp"),new Uint8Array(await montage.arrayBuffer()));
const inspect=await deck.inspect({kind:"slide,textbox,shape,image,chart,notes,layout",maxChars:20000}); await fs.writeFile(path.join(outDir,"inspect.ndjson"),inspect.ndjson,"utf8");
const pptx=await PresentationFile.exportPptx(deck); await pptx.save(path.join(ROOT,"docs","SLIDES_FINAL.pptx"));
console.log(`Built ${deck.slides.items.length} slides`);
