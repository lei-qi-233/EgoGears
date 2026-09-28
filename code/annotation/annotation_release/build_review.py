import json, re, hashlib
from pathlib import Path
b=Path('/home/yuedong_tan/datasets')
out=b/'seg600_verification/annotation_app'
out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (b/'seg600_verification/segments_700_questions_with_category.jsonl').read_text().splitlines() if x.strip()]
assert len({r['question_id'] for r in rows})==len(rows)
raw={p.stem:p for p in (b/'GoogleDrive_1wD2FkpkIPMCNciGNTyT-GU-mbKOIOKHz').rglob('*.mp4')}
media={}; cases=[]; missing=[]
for r in rows:
 q=r['question_id']; start=float(r['start_second']); end=float(r['end_second'])
 assert end>start>=0 and set(r['answer'])<=set(r['options'])
 source=raw.get(r['video_id'])
 if source:
  key=hashlib.sha256(str(source).encode()).hexdigest()[:24]
  media[key]=str(source)
 else: missing.append({'question_id':q,'video_id':r['video_id']})
 cases.append(dict(review_id=q,source=r['video_id'],unit=r['segment'],question_type=r['category'],question=r['question'],options=r['options'],n_select=len(r['answer']),clips=[dict(label='视频片段',path=f'/media/{key}#t={start},{end}',start=start,end=end)] if source else []))
template=(b/'review.html').read_text()
template=re.sub(r'const CASES = .*?;\s*const KEY = .*?;',lambda m:'const CASES = '+json.dumps(cases,ensure_ascii=False).replace('</','<\\/')+';\nconst KEY = "rb_annot_seg698_v1";',template,flags=re.S)
template=template.replace('第 04 份','698 题复核').replace('bundle_04','seg698_v1')
template=template.replace('const KEY = "rb_annot_seg698_v1";', 'const KEY = "rb_annot_seg698_v1" + (location.hash === "#qa" ? "_qa" : "");')
template=template.replace('CLIP 标签顺序 ≠ 时间顺序。','播放器限定在题目对应的原视频时间段。')
template=template.replace('<div id="cards"></div>','<div class="row"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><div id="cards"></div>')
template=template.replace('function render(){','let page = 0; const PAGE_SIZE=20;\nfunction render(){')
template=template.replace('for (const c of CASES) {','const filtered=CASES.filter(c => f === "all" || (f === "done" ? isDone(c) : !isDone(c)));\n  page=Math.max(0,Math.min(page,Math.ceil(filtered.length/PAGE_SIZE)-1));\n  document.getElementById("page").textContent=`第 ${page+1} / ${Math.max(1,Math.ceil(filtered.length/PAGE_SIZE))} 页`;\n  for (const c of filtered.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE)) {')
template=template.replace('box.appendChild(d);','box.appendChild(d);\n    if (!c.clips.length) { const warning=document.createElement("p"); warning.className="warn"; warning.textContent="缺少原视频："+c.source+"。请勿凭文字猜测答案，可记录 cant_tell。"; d.querySelector(".clips").appendChild(warning); }\n    d.querySelectorAll("video").forEach((v,i)=>{const k=c.clips[i];v.addEventListener("loadedmetadata",()=>{v.currentTime=k.start;});v.addEventListener("play",()=>{if(v.currentTime>=k.end || v.currentTime<k.start)v.currentTime=k.start;});v.addEventListener("timeupdate",()=>{if(v.currentTime>=k.end)v.pause();});v.addEventListener("error",()=>{v.parentElement.querySelector(".lbl").textContent="视频加载失败，请标记 cant_tell 并记录原因";});});')
template=template.replace('render();\n</script>','document.getElementById("prev").onclick=()=>{page--;render();window.scrollTo(0,0);};\ndocument.getElementById("next").onclick=()=>{page++;render();window.scrollTo(0,0);};\nrender();\n</script>')
template=template.replace('source: c.source, unit: c.unit, question_type: c.question_type, annotator:', 'source: c.source, unit: c.unit, question_type: c.question_type, media_available: !!c.clips.length, annotator:')
template=template.replace('请勿凭文字猜测答案，可记录 cant_tell。','请先跳过此题，等待补齐视频；不要猜测答案。')
(out/'index.html').write_text(template)
(out/'media.json').write_text(json.dumps(media))
report={'questions':len(rows),'available_questions':len(rows)-len(missing),'available_videos':len(media),'missing':missing}
(out/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='missing'}))
