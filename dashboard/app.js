'use strict';
const $ = s => document.querySelector(s);
const escapeHTML = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const E = escapeHTML;
const names = {imgColorWheel:'Hue rotation',imgDarkBright:'Brightness',imgMotionBlur:'Motion blur',imgResolution:'Resolution',imgStretch:'Stretch',imgGrayscale:'Grayscale',blue_to_orange:'Blue → orange',blue_to_yellow:'Blue → yellow',yellow_to_blue:'Yellow → blue',yellow_to_orange:'Yellow → orange',original_vs_transformed:'Detection consistency',transformed_vs_gt:'Accuracy against ground truth',original_vs_transformed_consistency:'Detection consistency',gt_accuracy_degradation:'Accuracy degradation',brute_force:'Brute force',golden_section:'Golden section'};
const nice = v => names[v] || String(v).replace(/^yolo/, 'YOLO').replaceAll('_', ' ');
const num = (v, digits=3) => Number.isFinite(v) ? v.toFixed(digits) : '—';
const pct = v => Number.isFinite(v) ? `${(100*v).toFixed(1)}%` : '—';
const unique = a => [...new Set(a)].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
const valid = Number.isFinite;
let data, request=0;
const state = {source:'paper',rq:'1',dataset:'',validationDataset:'',metric:'AP',comparison:'original_vs_transformed',rule:'imgDarkBright',evidence:'response',stage:'search',model:'',method:'brute_force',analysis:'original_vs_transformed_consistency'};
function select(id, title, values, selected, labels={}) {
  return `<label>${E(title)}<select id="${id}">${values.map(v=>`<option value="${E(v)}" ${v===selected?'selected':''}>${E(labels[v] || nice(v))}</option>`).join('')}</select></label>`;
}
function bind(id,key){const el=$('#'+id);if(el)el.onchange=()=>{state[key]=el.value;render();};}
function keep(key,values){if(!values.includes(state[key]))state[key]=values[0] || '';}
function cards(items){return `<div class="cards">${items.map(([value,label,detail])=>`<div class="card"><b>${E(value)}</b><span>${E(label)}</span>${detail?`<small>${E(detail)}</small>`:''}</div>`).join('')}</div>`;}
function empty(title,description){return `<div class="empty"><h2>${E(title)}</h2>${E(description)}</div>`;}
function url(path){return '/results/'+state.source+'/'+path.split('/').map(encodeURIComponent).join('/');}
function figure(path,title){if(!data.figures.includes(path))return empty('Figure not generated', 'The selected results do not include this visualisation.');return `<figure class="figure"><a href="${url(path)}" target="_blank" rel="noopener" aria-label="Open ${E(title)} at full resolution"><img src="${url(path)}" alt="${E(title)}" loading="lazy"></a><figcaption><span>${E(title)}</span><a href="${url(path)}" target="_blank" rel="noopener">Open full size ↗</a></figcaption></figure>`;}
function bars(rows,max=1){return `<div class="bars">${rows.map(([label,v])=>`<div class="bar-row"><span>${E(label)}</span><div class="track"><div class="fill" style="width:${valid(v)?Math.min(100,Math.max(0,100*v/max)):0}%"></div></div><span>${num(v)}</span></div>`).join('')}</div>`;}
function tabs(items,key){return `<div class="tabs">${Object.entries(items).map(([k,label])=>`<button data-key="${key}" data-value="${k}" class="${state[key]===k?'active':''}" aria-pressed="${state[key]===k}">${E(label)}</button>`).join('')}</div>`;}
function intro(number,title,text){return `<p class="eyebrow">RESEARCH QUESTION ${number}</p><h1>${title}</h1><p class="intro">${text}</p>`;}
function rq1(){
  const datasets=Object.keys(data.datasets); keep('dataset',datasets);
  let html=intro('01','Where does robustness break down?','Compare model behaviour under different transformations.');
  if(!datasets.length)return html+empty('No robustness results yet','Run the RQ1 analysis pipeline. Results will appear here after refresh.');
  const ds=data.datasets[state.dataset], rows=ds.rows;
  const comparisons=unique(rows.map(r=>r.comparison));keep('comparison',comparisons);
  const selected=rows.filter(r=>r.comparison===state.comparison);
  const rules=unique(selected.map(r=>r.rule));keep('rule',rules);
  const models=unique(selected.map(r=>r.model));
  html+=`<div class="filters">${select('dataset','Dataset',datasets,state.dataset)}${select('comparison','Compare',comparisons,state.comparison)}${select('metric','Metric',['AP','AR'],state.metric)}</div>`;
  const base=Object.entries(ds.baselines).map(([model,metrics])=>[nice(model),metrics[state.metric]]).sort((a,b)=>(b[1]??-1)-(a[1]??-1));
  if(base.length)html+=`<details><summary>Untransformed baseline · ${E(state.metric)} against ground truth</summary>${bars(base)}<p class="note">This dataset’s original-image accuracy; no external COCO benchmark scores are provided.</p></details>`;
  if(!rules.length)return html+empty('No transformation metrics yet','Baseline results are available above.');
  html+=`<section class="panel"><div class="panel-heading"><h2>Explore a transformation</h2><span class="badge">${E(nice(state.rule))}</span></div><div class="filters">${select('rule','Transformation',rules,state.rule)}</div>${tabs({response:'Parameter response',overlap:'Retained & lost detections',confidence:'Confidence shifts',colour:'Cone colours'},'evidence')}`;
  const prefix=`rq1/${state.dataset}/`;
  if(state.evidence==='response'){
    const path=prefix+`parameter_variations/${state.metric==='AP'?'average_precision':'average_recall'}/${state.rule}_${state.comparison}.png`;
    html+=data.figures.includes(path)?figure(path,`${nice(state.rule)} · ${nice(state.comparison)} · ${state.metric}`):responseChart(selected.filter(r=>r.rule===state.rule),models);
  } else if(state.evidence==='overlap')html+=`<p class="note">How true positives, false positives and new detections change after transformation. These rates use ground-truth matching.</p>`+figure(prefix+`detection_overlap_metrics/plots/${state.rule}_detection_overlap_metrics.png`,`${nice(state.rule)} · detection overlap`);
  else if(state.evidence==='confidence')html+=figure(prefix+`confidence_change/${state.rule}.png`,`${nice(state.rule)} · confidence changes`);
  else html+=`<p class="note">All cone-colour transformations, independent of the selected transformation.</p>`+figure(prefix+'cone_colour_analysis/cone_colour_comparison.png','Cone-colour comparison');
  html+='</section><p class="note">Failure-image overlap and train/test comparisons require their corresponding experiment outputs; aggregate detection rates alone cannot establish either.</p>';
  return html;
}
function responseChart(rows,models){
  if(!rows.length)return empty('No parameter results','No valid results are available for this selection.');
  const variants=unique(rows.map(r=>r.variant));
  const colours=['#176b58','#537dae','#c78c36','#ab6289','#6d65a2'];
  const x=i=>65+i*650/Math.max(1,variants.length-1),y=v=>245-v*205;
  let svg=`<svg viewBox="0 0 800 330" role="img" aria-label="${E(state.metric)} by transformation setting"><title>${E(state.metric)} by transformation setting</title>`;
  for(const v of [0,.25,.5,.75,1])svg+=`<line x1="65" x2="730" y1="${y(v)}" y2="${y(v)}" stroke="#e4e9e3"/><text x="48" y="${y(v)+4}" text-anchor="end" class="chart-label">${v}</text>`;
  variants.forEach((v,i)=>{const setting=rows.find(r=>r.variant===v).setting;const label=Object.keys(setting).length?Object.entries(setting).map(([k,x])=>`${k}=${x}`).join(', '):v;svg+=`<text x="${x(i)}" y="267" text-anchor="middle" class="chart-label"><title>${E(label)}</title>${i+1}</text>`;});
  models.forEach((m,mi)=>{let previous=null;variants.forEach((v,i)=>{const r=rows.find(r=>r.model===m&&r.variant===v);if(!r||!valid(r[state.metric])){previous=null;return;}const point=[x(i),y(r[state.metric])];if(previous)svg+=`<line x1="${previous[0]}" y1="${previous[1]}" x2="${point[0]}" y2="${point[1]}" stroke="${colours[mi%5]}" stroke-width="2"/>`;svg+=`<circle cx="${point[0]}" cy="${point[1]}" r="4" fill="${colours[mi%5]}"><title>${E(nice(m))} · ${E(v)} · ${num(r[state.metric])}</title></circle>`;previous=point;});});
  svg+='<text x="390" y="300" text-anchor="middle" class="chart-label">Setting index (parameter values below)</text></svg>';
  return svg+`<div class="legend">${models.map((m,i)=>`<span><i style="background:${colours[i%5]}"></i>${E(nice(m))}</span>`).join('')}</div><details><summary>Setting labels</summary>${variants.map((v,i)=>`<p>${i+1} · ${E(v)} · ${E(JSON.stringify(rows.find(r=>r.variant===v).setting))}</p>`).join('')}</details>`;
}
function rq2(){
  let html=intro('02','Can we find brightness failures faster?','Follow the evidence from search efficiency to brightness scoring and the resulting accuracy distribution.');
  html+=tabs({search:'2a · Search efficiency',scoring:'2b · Brightness scoring',distribution:'2c · Accuracy distribution'},'stage');
  let figures=data.figures.filter(p=>p.startsWith('rq2/'));
  const models=state.stage==='search'?unique([...Object.keys(data.comparisons),...Object.keys(data.search)]):state.stage==='scoring'?unique([...Object.values(data.validation).flatMap(group=>Object.keys(group)),...figures.filter(p=>p.includes('/brightness_scoring/')).map(p=>p.split('/')[4])]):unique(figures.filter(p=>p.includes('/performance_distributions/')).map(p=>p.split('/').pop().replace(/_(brute_force|golden_section)_performance_distribution\.png$/,'')));
  keep('model',models);
  if(!models.length)return html+empty('No results for this question yet','Generate this analysis stage, then refresh to explore its results.');
  html+=`<div class="filters">${select('model','Model',models,state.model)}`;
  if(state.stage!=='search'){
    const methods=state.stage==='scoring'?unique(figures.filter(p=>p.includes('/brightness_scoring/')&&p.split('/')[4]===state.model).map(p=>p.split('/')[3])):unique(figures.filter(p=>p.includes('/performance_distributions/')&&p.includes('/'+state.model+'_')).map(p=>p.includes('_brute_force_')?'brute_force':'golden_section'));
    keep('method',methods);if(methods.length)html+=select('method','Search method',methods,state.method);
  }
  html+='</div>';
  if(state.stage==='search'){
    const c=data.comparisons[state.model];
    if(c){
      html+=cards([[pct(c.avg_iteration_reduction),'Mean iteration reduction','Golden section vs brute force'],[num(c.avg_speedup,2)+'×','Mean per-image speedup',`${c.num_images ?? '—'} matched images`],[`${num(c.avg_gamma_left_diff)} / ${num(c.avg_gamma_right_diff)}`,'Mean |Δγ| · bright / dark','Threshold agreement with brute force']]);
      html+=`<section class="panel"><h2>Search cost and threshold agreement</h2><p class="note">Speed and agreement answer different questions: fewer iterations do not guarantee the same operational range.</p>${figure(`rq2/rq2a/comparison/${state.model}/comparison.png`,`${nice(state.model)} · brute force vs golden section`)}</section>`;
    }else html+=empty('No paired comparison yet','Search outputs exist, but this model does not yet have a generated comparison.');
    const runs=data.search[state.model]||{};
    html+=`<details><summary>Search coverage & source</summary>${Object.entries(runs).map(([method,r])=>`<p>${E(nice(method))} · ${E(r.dataset)} · target AP ${num(r.target_ap,2)} · ${E(r.processed_images ?? '—')} processed · ${E(r.skipped_images ?? '—')} skipped · ${E(r.failed_images ?? '—')} failed. <a href="${url(`rq2/rq2a/${method}/${state.model}/results.json`)}" target="_blank" rel="noopener">Results ↗</a></p>`).join('')}<p>Comparison statistics come from the saved matched-image summary, whose coverage can differ from the complete search runs.</p></details>`;
  }else if(state.stage==='scoring'){
    const validationDatasets=Object.keys(data.validation).filter(dataset=>data.validation[dataset][state.model]);
    keep('validationDataset',validationDatasets);
    if(validationDatasets.length)html+=`<div class="filters">${select('validationDataset','Validation dataset',validationDatasets,state.validationDataset)}</div>`;
    const v=data.validation[state.validationDataset]?.[state.model];
    if(v){
      const metrics=Object.entries(v.metrics).filter(([,x])=>valid(x.balanced_accuracy)).sort((a,b)=>b[1].balanced_accuracy-a[1].balanced_accuracy);
      const best=metrics[0];
      html+=cards([[best?nice(best[0]):'—','Highest balanced accuracy','Among methods validated for this model'],[best?pct(best[1].balanced_accuracy):'—','Balanced accuracy','Mean failure recall and original acceptance'],[v.holdout_summary.eligible_originals ?? '—','Eligible holdout originals',`${v.holdout_summary.selected_transformed_failures ?? '—'} transformed failures`]]);
      html+=`<section class="panel"><h2>Does the score identify failures?</h2><p class="note">Holdout validation · failure threshold AP ${num(v.ap_threshold,2)}. Validation uses its saved operating ranges and is independent of the search-method selector.</p>${figure(v.figure,`${nice(state.model)} · brightness validation`)}</section>`;
    }else html+=`<p class="warning">No holdout validation is available for this model. Boundary-score variation alone does not establish predictive reliability.</p>`;
    html+=`<section class="panel"><h2>Which scores are consistent at the boundaries?</h2><p class="note">Compare the spread at bright and dark failure thresholds. Scores can use different scales.</p>${figure(`rq2/rq2b/brightness_scoring/${state.method}/${state.model}/method_comparison.png`,`${nice(state.model)} · ${nice(state.method)} · brightness scores`)}</section>`;
  }else {
    const distributionPath=figures.find(p=>p.includes('/performance_distributions/')&&p.endsWith(`/${state.model}_${state.method}_performance_distribution.png`));
    html+=`<section class="panel"><h2>How does accuracy shift at the boundaries?</h2><p class="note">Original images versus the bright and dark operational boundaries, for the selected search method.</p>${figure(distributionPath,`${nice(state.model)} · ${nice(state.method)} · accuracy distribution`)}</section>`;
  }
  return html;
}
function rq3(){
  let html=intro('03','Can a smaller sample tell the same story?','Check agreement with the full dataset, including the uncertainty around each comparison.');
  const r=data.rq3;if(!r.metadata)return html+empty('No sample-equivalence results yet','Run the RQ3 analysis, then refresh to see sample sizes and equivalence results.');
  const m=r.metadata;
  const analyses=Object.keys(m.analyses||{});keep('analysis',analyses);
  const metrics=m.metrics||['AP','AR'];keep('metric',metrics);
  const all=Array.isArray(r.parameter_results)?r.parameter_results:[];
  const rows=all.filter(x=>x.analysis===state.analysis&&x.metric===state.metric);
  const rules=unique(rows.map(x=>x.rule));keep('rule',rules);
  html+=`<div class="filters">${select('analysis','Measure',analyses,state.analysis)}${select('metric','Metric',metrics,state.metric)}</div>`;
  const eq=rows.filter(x=>x.classification==='equivalent').length;
  const auc=(Array.isArray(r.auc_results)?r.auc_results:[]).filter(x=>x.analysis===state.analysis&&x.metric===state.metric);
  const arms=Object.entries(m.arms||{}).map(([arm,x])=>`${arm}: ${x.image_count} images`).join(' · ');
  html+=cards([[pct(m.sample_fraction),'Dataset sampled',arms],[rows.length?pct(eq/rows.length):'—','Settings equivalent',`${eq} of ${rows.length} model × setting comparisons`],[`${auc.filter(x=>x.classification==='equivalent').length} / ${auc.length}`,'Response curves equivalent','Normalised area under each parameter curve']]);
  html+=`<section class="panel"><h2>Agreement by transformation</h2><p class="note">${pct(m.confidence_interval?.level)} confidence intervals · equivalence margin ±${num(m.equivalence_margin,2)} · ${E(m.bootstrap_repetitions)} bootstrap repetitions. Inconclusive does not mean different.</p>`;
  html+=rules.map(rule=>{const group=rows.filter(x=>x.rule===rule);return `<div class="bar-row" style="grid-template-columns:120px 1fr 90px;margin:16px 0"><span>${E(nice(rule))}</span><div class="stack" role="img" aria-label="${E(nice(rule))} equivalence outcomes">${[['equivalent','eq'],['inconclusive','inc'],['meaningfully_different','diff']].map(([c,cls])=>{const count=group.filter(x=>x.classification===c).length;return `<span class="${cls}" style="width:${100*count/group.length}%" title="${E(nice(c))}: ${count} of ${group.length}"></span>`;}).join('')}</div><span>${group.filter(x=>x.classification==='equivalent').length}/${group.length} equiv.</span></div>`;}).join('');
  html+=`<div class="legend"><span><i class="eq"></i>Equivalent</span><span><i class="inc"></i>Inconclusive</span><span><i class="diff"></i>Meaningfully different</span></div><p class="note">${E(m.analyses?.[state.analysis]?.statistic_description||'')} Grayscale has a single setting, so no curve-area comparison is defined.</p></section>`;
  html+=`<section class="panel"><h2>Full dataset versus sample</h2><div class="filters">${select('rule','Transformation',rules,state.rule)}</div>${figure(`rq3/sample_equivalence/plots/${state.rule}_${state.metric.toLowerCase()}_${state.analysis==='original_vs_transformed_consistency'?'original_vs_transformed':'gt_accuracy_degradation'}.png`,`${nice(state.rule)} · ${nice(state.analysis)} · ${state.metric} with confidence intervals`)}</section>`;
  const v=r.validation;
  if(v?.values_different_from_stored_json)html+=`<p class="warning">Validation note: ${E(v.values_different_from_stored_json)} of ${E(v.values_checked)} recomputed full-dataset values differ from stored results (maximum ${num(v.maximum_absolute_difference,4)}). Displayed full-dataset points use the stored values.</p>`;
  html+=`<details><summary>Study scope & source</summary><p>Dataset: ${E(m.source_dataset)}. Evidence applies to the saved sample and configured transformations. This analysis does not establish test subsumption.</p><p><a href="${url('rq3/sample_equivalence/report.md')}" target="_blank" rel="noopener">Analysis report ↗</a> · <a href="${url('rq3/sample_equivalence/results.json')}" target="_blank" rel="noopener">Source results ↗</a></p></details>`;
  return html;
}
function render(){
  if(!data)return;
  $('#main').innerHTML=({'1':rq1,'2':rq2,'3':rq3}[state.rq])()+(data.warnings.length?`<details><summary>${data.warnings.length} unreadable result file(s)</summary>${data.warnings.map(x=>`<p>${E(x)}</p>`).join('')}</details>`:'');
  for(const key of ['dataset','validationDataset','metric','comparison','rule','model','method','analysis'])bind(key,key);
  document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{state[b.dataset.key]=b.dataset.value;render();});
  document.querySelectorAll('nav button').forEach(b=>{b.classList.toggle('active',b.dataset.rq===state.rq);b.setAttribute('aria-pressed',String(b.dataset.rq===state.rq));});
}
async function load(){
  const current=++request;data=null;
  $('#main').innerHTML='<p class="empty">Loading results…</p>';
  $('#provenance').textContent='';
  try{
    const response=await fetch('/api/'+state.source);if(!response.ok)throw new Error('The results could not be loaded. Check the server and refresh.');
    const result=await response.json();if(current!==request)return;
    data=result;$('#provenance').textContent=data.root+' · loaded '+new Date().toLocaleTimeString();render();
  }catch(error){if(current===request)$('#main').innerHTML=empty('Unable to load results',error.message);}
}
$('#refresh').onclick=load;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{state.rq=b.dataset.rq;render();});
load();
