import { useState, useCallback, useRef, useEffect } from "react";
import AdvisorDashboard from "./AdvisorDashboard.jsx";
import AdminDashboard from "./AdminDashboard.jsx";
import StudentDashboard from "./StudentDashboard.jsx";
import { apiFetch as fetchApi } from "./apiClient.js";

// ── STYLES ─────────────────────────────────────────────────────────────────
const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Instrument+Serif:ital@0;1&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#ffffff;
  --bg2:#f8f9fc;
  --bg3:#f2f4f8;
  --surface:#ffffff;
  --surface2:#eef1f6;
  --border:#d8deea;
  --border2:#c8d1e0;
  --text:#1a2233;
  --text2:#3d4a63;
  --text3:#667691;
  --accent:#4f7cff;
  --accent2:#6b8fff;
  --accent-dim:#4f7cff14;
  --accent-glow:rgba(79,124,255,0.2);
  --green:#2da66f;
  --green-dim:#2da66f12;
  --amber:#c58a2a;
  --amber-dim:#c58a2a12;
  --red:#d16a6a;
  --red-dim:#d16a6a12;
  --teal:#2c9fb0;
  --teal-dim:#2c9fb012;
  --pink:#c97cb6;
  --pink-dim:#c97cb612;
  --r:8px;--r2:14px;--r3:20px;
  --mono:'DM Mono',monospace;
  --sans:'Syne',sans-serif;
  --serif:'Instrument Serif',serif;
  --shadow:0 8px 24px rgba(15,30,60,0.08);
  --shadow2:0 14px 38px rgba(15,30,60,0.12);
}
html,body,#root{height:100%;background:var(--bg);}
body{font-family:var(--sans);font-size:14px;line-height:1.6;color:var(--text);-webkit-font-smoothing:antialiased;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:2px;}

/* LAYOUT */
.app{min-height:100vh;display:flex;flex-direction:column;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:.875rem 2rem;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.88);backdrop-filter:blur(20px);position:sticky;top:0;z-index:200;}
.logo{display:flex;align-items:center;gap:.65rem;font-family:var(--sans);font-size:1.1rem;font-weight:800;letter-spacing:-.5px;color:var(--text);}
.logo-icon{width:32px;height:32px;background:linear-gradient(135deg,var(--accent),var(--pink));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;box-shadow:0 0 20px var(--accent-glow);}
.logo-sub{font-family:var(--mono);font-size:.65rem;color:var(--text3);font-weight:400;letter-spacing:1px;text-transform:uppercase;}
.main{flex:1;max-width:960px;width:100%;margin:0 auto;padding:2.5rem 1.5rem 6rem;}

/* HERO */
.hero{margin-bottom:2.5rem;padding-bottom:2rem;border-bottom:1px solid var(--border);}
.hero-eyebrow{font-family:var(--mono);font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;}
.hero-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.5;transform:scale(.8);}}
.hero h1{font-family:var(--serif);font-size:clamp(2rem,5vw,3.2rem);font-weight:400;line-height:1.05;color:var(--text);margin-bottom:.5rem;}
.hero h1 em{font-style:italic;color:var(--accent2);}
.hero-desc{font-family:var(--mono);font-size:.72rem;color:var(--text3);display:flex;flex-wrap:wrap;gap:1rem;margin-top:.75rem;}
.hero-tag{display:inline-flex;align-items:center;gap:.35rem;}
.hero-tag::before{content:'◆';color:var(--accent);font-size:.5rem;}

/* STEPS */
.step-rail{display:flex;gap:0;margin-bottom:2.5rem;position:relative;}
.step-rail::before{content:'';position:absolute;top:14px;left:0;right:0;height:1px;background:var(--border2);z-index:0;}
.step-seg{flex:1;display:flex;flex-direction:column;align-items:center;gap:.4rem;position:relative;z-index:1;}
.step-circle{width:29px;height:29px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:.68rem;font-weight:600;border:1px solid var(--border2);background:var(--bg2);color:var(--text3);transition:all .25s;}
.step-seg.done .step-circle{background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:0 0 12px var(--accent-glow);}
.step-seg.active .step-circle{background:var(--bg2);border-color:var(--accent);color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim);}
.step-name{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--text3);}
.step-seg.active .step-name{color:var(--accent);}
.step-seg.done .step-name{color:var(--accent2);}

/* PANEL */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--r2);padding:1.75rem;margin-bottom:1.25rem;box-shadow:var(--shadow);}
.panel-title{font-family:var(--sans);font-size:1rem;font-weight:700;color:var(--text);margin-bottom:.2rem;letter-spacing:-.3px;}
.panel-sub{font-family:var(--mono);font-size:.68rem;color:var(--text3);margin-bottom:1.5rem;}

/* FORMS */
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
@media(max-width:580px){.fgrid{grid-template-columns:1fr;}}
.field{display:flex;flex-direction:column;gap:.35rem;}
.flabel{font-family:var(--mono);font-size:.62rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--text3);}
.flabel .req{color:var(--accent);}
input,select,textarea{background:var(--bg3);border:1px solid var(--border2);color:var(--text);border-radius:var(--r);padding:.6rem .875rem;font-family:var(--sans);font-size:.88rem;width:100%;outline:none;transition:border-color .2s,box-shadow .2s;-webkit-appearance:none;}
input:focus,select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim);}
input::placeholder{color:var(--text3);}
select option{background:var(--bg3);}
input[type=range]{-webkit-appearance:none;height:4px;background:var(--border2);border-radius:2px;padding:0;border:none;box-shadow:none;cursor:pointer;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-dim);cursor:pointer;}
input[type=range]:focus{box-shadow:none;border:none;}
input[type=checkbox]{width:15px;height:15px;accent-color:var(--accent);cursor:pointer;border:1px solid var(--border2);}

/* BUTTONS */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:.4rem;padding:.6rem 1.4rem;border-radius:var(--r);font-family:var(--sans);font-size:.88rem;font-weight:700;letter-spacing:-.2px;cursor:pointer;border:none;transition:all .15s;}
.btn-primary{background:var(--accent);color:#fff;box-shadow:0 4px 20px var(--accent-glow);}
.btn-primary:hover{background:var(--accent2);transform:translateY(-1px);box-shadow:0 6px 28px var(--accent-glow);}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none;}
.btn-ghost{background:var(--surface2);color:var(--text2);border:1px solid var(--border2);}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent);}
.btn-danger{background:var(--red-dim);color:var(--red);border:1px solid #f5656522;}
.btn-sm{padding:.35rem .875rem;font-size:.78rem;}
.btn-full{width:100%;}

/* ALERTS */
.alert{display:flex;gap:.6rem;padding:.75rem 1rem;border-radius:var(--r);font-size:.82rem;margin:.5rem 0;font-family:var(--mono);}
.alert-info{background:var(--teal-dim);border:1px solid #22d3ee25;color:var(--teal);}
.alert-warn{background:var(--amber-dim);border:1px solid #f5a62325;color:var(--amber);}
.alert-error{background:var(--red-dim);border:1px solid #f5656525;color:var(--red);}
.alert-ok{background:var(--green-dim);border:1px solid #3ecf8e25;color:var(--green);}
.alert-ai{background:linear-gradient(135deg,var(--accent-dim),var(--pink-dim));border:1px solid #7c6ef730;color:var(--accent2);}

/* SECTION LABELS */
.slabel{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:2px;color:var(--text3);margin:1.5rem 0 .75rem;display:flex;align-items:center;gap:.6rem;}
.slabel::after{content:'';flex:1;height:1px;background:var(--border);}

/* TABS */
.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:1.75rem;}
.tab{background:none;border:none;color:var(--text3);font-family:var(--sans);font-size:.88rem;font-weight:600;letter-spacing:-.2px;padding:.6rem 1.25rem;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .2s;}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tab:hover:not(.active){color:var(--text2);}

/* AI EXTRACTION SPECIFIC */
.ai-status-bar{display:flex;align-items:center;gap:.75rem;padding:.75rem 1rem;border-radius:var(--r);background:linear-gradient(135deg,var(--accent-dim),var(--pink-dim));border:1px solid #7c6ef730;margin-bottom:1rem;}
.ai-pulse{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 0 var(--accent-glow);animation:aiPulse 1.5s infinite;}
@keyframes aiPulse{0%{box-shadow:0 0 0 0 var(--accent-glow);}70%{box-shadow:0 0 0 8px rgba(79,124,255,0);}100%{box-shadow:0 0 0 0 rgba(79,124,255,0);}}
.ai-status-text{font-family:var(--mono);font-size:.72rem;color:var(--accent2);flex:1;}
.ai-conf-badge{font-family:var(--mono);font-size:.65rem;padding:.2rem .6rem;border-radius:4px;font-weight:600;}
.conf-high{background:var(--green-dim);color:var(--green);border:1px solid #3ecf8e33;}
.conf-mid{background:var(--amber-dim);color:var(--amber);border:1px solid #f5a62333;}
.conf-low{background:var(--red-dim);color:var(--red);border:1px solid #f5656533;}

/* DOC TYPE CARDS — updated for 9-type tab grid */
.doctype-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin-bottom:1.25rem;}
.doctype-card{padding:.65rem .5rem;border:1px solid var(--border);border-radius:var(--r);background:var(--bg3);cursor:pointer;text-align:center;transition:all .2s;position:relative;}
.doctype-card:hover{border-color:var(--border2);}
.doctype-card.selected{border-color:var(--accent);background:var(--accent-dim);box-shadow:0 0 0 1px var(--accent);}
.doctype-card.uploaded{border-color:#3ecf8e55;background:var(--green-dim);}
.doctype-card.uploaded.selected{border-color:var(--green);box-shadow:0 0 0 1px var(--green);}
.doctype-icon{font-size:1.25rem;margin-bottom:.3rem;}
.doctype-name{font-family:var(--mono);font-size:.6rem;font-weight:500;color:var(--text3);letter-spacing:.5px;}
.doctype-card.selected .doctype-name{color:var(--accent2);}
.doctype-card.uploaded .doctype-name{color:var(--green);}
.doctype-done-dot{position:absolute;top:4px;right:4px;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);}

/* UPLOAD */
.upload-zone{border:2px dashed var(--border2);border-radius:var(--r2);padding:2.5rem;text-align:center;cursor:pointer;transition:all .2s;background:var(--bg3);}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:var(--accent-dim);}
.upload-icon{font-size:2.2rem;margin-bottom:.75rem;opacity:.7;}
.upload-title{font-family:var(--sans);font-size:1rem;font-weight:700;margin-bottom:.3rem;}
.upload-sub{font-family:var(--mono);font-size:.68rem;color:var(--text3);}
.upload-preview{width:100%;max-height:280px;object-fit:contain;border-radius:var(--r);margin:1rem 0;border:1px solid var(--border);}

/* PIPELINE */
.pipeline{display:flex;flex-direction:column;gap:.4rem;margin:1rem 0;}
.pipe-step{display:flex;align-items:center;gap:.75rem;font-family:var(--mono);font-size:.72rem;padding:.5rem .75rem;border-radius:var(--r);background:var(--bg3);border:1px solid var(--border);transition:all .2s;}
.pipe-icon{font-size:.85rem;width:20px;text-align:center;}
.pipe-label{flex:1;color:var(--text2);}
.pipe-status{font-size:.65rem;font-weight:600;}
.ps-wait{color:var(--text3);}
.ps-run{color:var(--amber);animation:pulse 1s infinite;}
.ps-ok{color:var(--green);}
.ps-err{color:var(--red);}
.pipe-step.ps-active{border-color:var(--accent);background:var(--accent-dim);}
.pipe-step.ps-done{border-color:#3ecf8e22;background:var(--green-dim);}

/* DATA DISPLAY */
.data-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:1rem 0;}
@media(max-width:500px){.data-grid{grid-template-columns:1fr;}}
.data-cell{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:.65rem .875rem;}
.data-label{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:.25rem;}
.data-val{font-size:.88rem;font-weight:500;color:var(--text);}
.subjects-table{width:100%;border-collapse:collapse;margin:1rem 0;}
.subjects-table th{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);text-align:left;padding:.5rem .75rem;border-bottom:1px solid var(--border);}
.subjects-table td{padding:.5rem .75rem;border-bottom:1px solid var(--border);font-size:.85rem;}
.grade-chip{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-family:var(--mono);font-size:.75rem;font-weight:700;}
.gchip-A{background:var(--green-dim);color:var(--green);}
.gchip-B{background:var(--teal-dim);color:var(--teal);}
.gchip-C{background:var(--accent-dim);color:var(--accent2);}
.gchip-S{background:var(--amber-dim);color:var(--amber);}
.gchip-F{background:var(--red-dim);color:var(--red);}
.score-section-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem;margin:.75rem 0;}
.score-section-cell{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:.75rem;}
.score-section-label{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:.4rem;}
.score-section-val{font-family:var(--mono);font-size:1.4rem;font-weight:300;color:var(--accent2);}
.score-section-bar{height:3px;background:var(--border);border-radius:2px;margin-top:.4rem;overflow:hidden;}
.score-section-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--pink));}

/* ELIGIBILITY */
.elig-hero{text-align:center;padding:2.5rem 2rem;margin-bottom:1.5rem;border-radius:var(--r2);border:1px solid;}
.elig-hero.pass{background:linear-gradient(135deg,var(--accent-dim),var(--green-dim));border-color:#7c6ef733;}
.elig-hero.fail{background:linear-gradient(135deg,var(--amber-dim),var(--red-dim));border-color:#f5a62333;}
.elig-score{font-family:var(--mono);font-size:3.5rem;font-weight:300;letter-spacing:-3px;line-height:1;}
.pass .elig-score{color:var(--accent2);}
.fail .elig-score{color:var(--amber);}
.elig-tier{display:inline-block;margin-top:.75rem;font-family:var(--mono);font-size:.65rem;letter-spacing:2px;text-transform:uppercase;padding:.25rem .875rem;border-radius:4px;font-weight:600;}
.tier-top{background:var(--accent-dim);color:var(--accent2);border:1px solid var(--accent)33;}
.tier-good{background:var(--green-dim);color:var(--green);border:1px solid var(--green)33;}
.tier-average{background:var(--teal-dim);color:var(--teal);border:1px solid var(--teal)33;}
.tier-foundation{background:var(--amber-dim);color:var(--amber);border:1px solid var(--amber)33;}
.chips-row{display:flex;flex-wrap:wrap;gap:.4rem;margin:.65rem 0;}
.chip{padding:.25rem .65rem;border-radius:4px;font-family:var(--mono);font-size:.68rem;font-weight:600;}
.chip-green{background:var(--green-dim);color:var(--green);border:1px solid #3ecf8e22;}
.chip-blue{background:var(--teal-dim);color:var(--teal);border:1px solid #22d3ee22;}
.chip-amber{background:var(--amber-dim);color:var(--amber);border:1px solid #c58a2a22;}
.chip-red{background:var(--red-dim);color:var(--red);border:1px solid #d16a6a22;}

/* UNIVERSITIES */
.uni-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r2);padding:1.5rem;margin-bottom:1rem;transition:all .2s;}
.uni-card:hover{border-color:var(--border2);box-shadow:var(--shadow);}
.uni-header{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:.75rem;}
.uni-name{font-family:var(--sans);font-size:1rem;font-weight:700;color:var(--text);}
.uni-qs{font-family:var(--mono);font-size:.62rem;font-weight:600;background:var(--accent-dim);color:var(--accent2);border:1px solid #7c6ef722;padding:.18rem .55rem;border-radius:4px;white-space:nowrap;}
.uni-meta{display:flex;gap:1rem;flex-wrap:wrap;font-family:var(--mono);font-size:.68rem;color:var(--text3);margin-bottom:.75rem;}
.uni-tags{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.75rem;}
.uni-tag{background:var(--bg3);border:1px solid var(--border);border-radius:4px;font-size:.65rem;font-family:var(--mono);color:var(--text3);padding:.12rem .45rem;}
.uni-reqs{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:.75rem;}
.req-box{background:var(--bg3);border-radius:var(--r);padding:.5rem .75rem;}
.req-lbl{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:.15rem;}
.req-val{font-family:var(--mono);font-size:.85rem;font-weight:600;color:var(--accent2);}
.uni-link{font-family:var(--mono);font-size:.68rem;font-weight:600;color:var(--teal);text-decoration:none;display:inline-flex;align-items:center;gap:.3rem;}
.fin-tag{background:var(--green-dim);color:var(--green);border:1px solid #3ecf8e22;font-family:var(--mono);font-size:.65rem;padding:.18rem .5rem;border-radius:4px;}
.fin-warn-tag{background:var(--amber-dim);color:var(--amber);border:1px solid #f5a62322;font-family:var(--mono);font-size:.65rem;padding:.18rem .5rem;border-radius:4px;}

/* PROFILE TABS */
.profile-tabs{display:flex;gap:.4rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.ptab{padding:.4rem .875rem;border-radius:var(--r);font-family:var(--mono);font-size:.68rem;font-weight:500;letter-spacing:.5px;cursor:pointer;border:1px solid var(--border);background:var(--bg3);color:var(--text3);transition:all .2s;}
.ptab.active{background:var(--accent);border-color:var(--accent);color:#fff;}
.ptab:hover:not(.active){border-color:var(--border2);color:var(--text2);}

/* AUTH */
.auth-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1.5rem;background:var(--bg);background-image:radial-gradient(ellipse at 20% 50%,#7c6ef718 0%,transparent 55%),radial-gradient(ellipse at 80% 20%,#e879f912 0%,transparent 45%);}
.auth-card{background:var(--surface);border:1px solid var(--border2);border-radius:var(--r2);padding:2.5rem;width:100%;max-width:400px;box-shadow:var(--shadow2);}
.auth-logo{font-family:var(--sans);font-size:1.3rem;font-weight:800;display:flex;align-items:center;gap:.65rem;margin-bottom:.2rem;}
.auth-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--accent),var(--pink));border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 0 20px var(--accent-glow);}
.auth-sub{font-family:var(--mono);font-size:.65rem;color:var(--text3);margin-bottom:2rem;}

/* MISC */
.btn-row{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-top:1.75rem;}
.btn-row-end{justify-content:flex-end;}
.user-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r2);padding:1.1rem 1.5rem;display:flex;align-items:center;gap:1.25rem;margin-bottom:1.5rem;box-shadow:var(--shadow);}
.user-avatar{width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--pink));display:flex;align-items:center;justify-content:center;font-family:var(--sans);font-size:1.1rem;font-weight:800;color:#fff;flex-shrink:0;}
.user-name{font-family:var(--sans);font-size:.95rem;font-weight:700;}
.user-email{font-family:var(--mono);font-size:.65rem;color:var(--text3);}
.user-tags{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.3rem;}
.user-tag{font-family:var(--mono);font-size:.58rem;padding:.12rem .45rem;border-radius:4px;background:var(--accent-dim);color:var(--accent2);border:1px solid #7c6ef722;}
.raw-text-box{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:.875rem;margin:.75rem 0;max-height:160px;overflow-y:auto;}
.raw-text-box pre{font-family:var(--mono);font-size:.68rem;color:var(--text2);white-space:pre-wrap;word-break:break-all;}
.spin{display:inline-block;width:13px;height:13px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.fade-up{animation:fadeUp .3s ease forwards;}

/* GRADE PILLS */
.grade-row{display:flex;gap:.3rem;flex-wrap:wrap;}
.gpill{min-width:40px;padding:.3rem .6rem;border-radius:var(--r);font-family:var(--mono);font-size:.75rem;font-weight:600;border:1px solid var(--border);background:var(--bg3);color:var(--text3);cursor:pointer;text-align:center;transition:all .15s;}
.gpill.gA{background:var(--green-dim);border-color:#3ecf8e55;color:var(--green);}
.gpill.gB{background:var(--teal-dim);border-color:#22d3ee55;color:var(--teal);}
.gpill.gC{background:var(--accent-dim);border-color:#7c6ef755;color:var(--accent2);}
.gpill.gS{background:var(--amber-dim);border-color:#f5a62355;color:var(--amber);}
.gpill.gF{background:var(--red-dim);border-color:#f5656555;color:var(--red);}

/* ENG */
.eng-panel{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1.1rem;margin-bottom:.65rem;}
.eng-badge{font-family:var(--mono);font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;padding:.18rem .55rem;border-radius:4px;}
.eb-ielts{background:var(--accent-dim);color:var(--accent2);border:1px solid #7c6ef733;}
.eb-toefl{background:var(--teal-dim);color:var(--teal);border:1px solid #22d3ee33;}
.eb-pte{background:var(--pink-dim);color:var(--pink);border:1px solid #e879f933;}
.cbox-row{display:flex;align-items:center;gap:.6rem;cursor:pointer;font-size:.88rem;}
.score-row{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:.7rem;color:var(--text3);margin-bottom:.35rem;}
.score-val{color:var(--accent2);font-weight:600;}
.track{height:3px;background:var(--border2);border-radius:2px;overflow:hidden;margin-bottom:.35rem;}
.fill{height:100%;border-radius:2px;transition:width .3s;}
.fill-ielts{background:linear-gradient(90deg,var(--accent),var(--accent2));}
.fill-toefl{background:linear-gradient(90deg,var(--teal),#06b6d4);}
.fill-pte{background:linear-gradient(90deg,var(--pink),#c026d3);}
.band-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin-top:.75rem;}
.band-cell{display:flex;flex-direction:column;gap:.3rem;}
.band-label{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);}
.band-input{text-align:center;font-family:var(--mono);font-size:1.05rem !important;font-weight:600;padding:.45rem .3rem !important;color:var(--accent2);}
.band-bar{height:3px;border-radius:2px;background:var(--border2);margin-top:.3rem;overflow:hidden;}
.band-fill{height:100%;border-radius:2px;transition:width .3s;}
.band-status{font-family:var(--mono);font-size:.6rem;font-weight:600;text-align:center;margin-top:.15rem;}

/* FIN */
.fin-summary-bar{background:linear-gradient(135deg,var(--green-dim),var(--accent-dim));border:1px solid #3ecf8e22;border-radius:var(--r2);padding:1.1rem 1.5rem;margin-bottom:1rem;display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center;}
.fin-sum-label{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:var(--text3);}
.fin-sum-val{font-family:var(--mono);font-size:1.05rem;font-weight:600;color:var(--green);}
.fin-sum-val.warn{color:var(--amber);}
.fin-panel{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1.1rem;margin-bottom:.65rem;}
.currency-select{display:flex;gap:.5rem;align-items:center;}
.currency-select select{width:80px!important;flex-shrink:0;}
.currency-select input{flex:1;}
.scholarship-row{display:flex;align-items:center;gap:.75rem;padding:.55rem .875rem;border:1px solid var(--border);border-radius:var(--r);background:var(--bg3);margin-bottom:.45rem;cursor:pointer;transition:border-color .2s;}
.scholarship-row:hover{border-color:var(--border2);}
.scholarship-label{font-size:.85rem;flex:1;}
.scholarship-badge{font-family:var(--mono);font-size:.6rem;padding:.12rem .45rem;border-radius:4px;background:var(--green-dim);color:var(--green);border:1px solid #3ecf8e22;white-space:nowrap;}

/* AI SPECIFIC */
.ai-extraction-box{background:linear-gradient(135deg,var(--bg2),var(--surface));border:1px solid var(--accent)33;border-radius:var(--r2);padding:1.5rem;margin:1rem 0;}
.ai-field{display:flex;flex-direction:column;gap:.2rem;padding:.6rem;border-radius:var(--r);background:var(--bg3);border:1px solid var(--border);}
.ai-field-label{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--text3);}
.ai-field-val{font-size:.88rem;font-weight:500;color:var(--text);}
.ai-badge{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--mono);font-size:.6rem;padding:.2rem .55rem;border-radius:4px;background:linear-gradient(135deg,var(--accent-dim),var(--pink-dim));border:1px solid #7c6ef730;color:var(--accent2);}

/* ── DOCUMENT CHECKLIST (NEW v6) ── */
.doc-checklist{background:var(--surface);border:1px solid var(--border);border-radius:var(--r2);padding:1.25rem 1.5rem;margin-bottom:1.25rem;box-shadow:var(--shadow);}
.doc-checklist-title{font-family:var(--sans);font-weight:700;font-size:.9rem;color:var(--text);margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;}
.checklist-progress{height:5px;background:var(--border2);border-radius:3px;margin-bottom:1rem;overflow:hidden;}
.checklist-progress-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--green));transition:width .5s ease;}
.checklist-cats{display:flex;flex-direction:column;gap:.6rem;}
.checklist-cat-label{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--text3);margin-bottom:.35rem;}
.checklist-chips{display:flex;flex-wrap:wrap;gap:.35rem;}
.checklist-chip{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--mono);font-size:.65rem;padding:.25rem .6rem;border-radius:5px;border:1px solid var(--border);background:var(--bg3);color:var(--text3);transition:all .3s;}
.checklist-chip.done{background:var(--green-dim);border-color:#3ecf8e44;color:var(--green);}
.checklist-remaining{margin-top:.75rem;padding:.6rem .875rem;background:var(--amber-dim);border:1px solid #f5a62322;border-radius:var(--r);font-family:var(--mono);font-size:.7rem;color:var(--amber);}
.checklist-complete{margin-top:.75rem;padding:.65rem 1rem;background:var(--green-dim);border:1px solid #3ecf8e22;border-radius:var(--r);font-family:var(--mono);font-size:.75rem;font-weight:600;color:var(--green);text-align:center;}

/* ── ENG-NEXT BANNER (NEW v6) ── */
.eng-next-banner{background:linear-gradient(135deg,var(--accent-dim),var(--teal-dim));border:1px solid #7c6ef730;border-radius:var(--r2);padding:1.1rem 1.5rem;margin-bottom:1.25rem;display:flex;gap:1rem;align-items:flex-start;}
.eng-next-icon{font-size:1.6rem;flex-shrink:0;margin-top:.1rem;}
.eng-next-title{font-family:var(--sans);font-weight:700;font-size:.92rem;color:var(--accent2);margin-bottom:.3rem;}
.eng-next-desc{font-family:var(--mono);font-size:.68rem;color:var(--text3);margin-bottom:.65rem;}
.eng-next-btns{display:flex;gap:.5rem;flex-wrap:wrap;}
.eng-next-btn{padding:.35rem .875rem;border-radius:var(--r);font-family:var(--mono);font-size:.68rem;font-weight:600;cursor:pointer;border:1px solid;transition:all .15s;}
.eng-next-btn-ielts{background:var(--accent-dim);border-color:var(--accent);color:var(--accent2);}
.eng-next-btn-toefl{background:var(--teal-dim);border-color:var(--teal);color:var(--teal);}
.eng-next-btn-pte{background:var(--pink-dim);border-color:var(--pink);color:var(--pink);}
.eng-next-btn:hover{filter:brightness(1.2);}

/* ── CHATBOT ── */
.chat-fab{
  position:fixed;bottom:1.75rem;right:1.75rem;z-index:9999;
  width:52px;height:52px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--pink));
  border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  font-size:1.35rem;
  box-shadow:0 4px 24px var(--accent-glow),0 0 0 0 var(--accent-glow);
  transition:transform .2s,box-shadow .2s;
  animation:fabPulse 3s ease-in-out infinite;
}
.chat-fab:hover{transform:scale(1.08);box-shadow:0 6px 32px var(--accent-glow);}
@keyframes fabPulse{0%,100%{box-shadow:0 4px 24px var(--accent-glow),0 0 0 0 rgba(79,124,255,0.35);}50%{box-shadow:0 4px 24px var(--accent-glow),0 0 0 10px rgba(79,124,255,0);}}
.chat-fab-badge{
  position:absolute;top:-3px;right:-3px;
  width:16px;height:16px;border-radius:50%;
  background:var(--green);border:2px solid var(--bg);
  font-family:var(--mono);font-size:.5rem;font-weight:700;color:#fff;
  display:flex;align-items:center;justify-content:center;
}
.chat-panel{
  position:fixed;bottom:0;right:0;
  width:380px;max-width:100vw;
  height:560px;max-height:85vh;
  background:var(--surface);
  border:1px solid var(--border2);
  border-radius:var(--r2) var(--r2) 0 0;
  box-shadow:var(--shadow2);
  display:flex;flex-direction:column;
  z-index:9998;
  transform:translateY(100%) scale(0.96);
  opacity:0;
  transform-origin:bottom right;
  transition:transform .3s cubic-bezier(.34,1.56,.64,1),opacity .25s ease;
  pointer-events:none;
}
.chat-panel.open{
  transform:translateY(0) scale(1);
  opacity:1;
  pointer-events:all;
}
.chat-panel.expanded{
  right:1rem;
  bottom:1rem;
  width:min(920px,calc(100vw - 2rem));
  height:min(82vh,760px);
  max-height:none;
  border-radius:var(--r2);
  transform-origin:bottom right;
}
.chat-header{
  display:flex;align-items:center;gap:.75rem;
  padding:.875rem 1.25rem;
  border-bottom:1px solid var(--border);
  background:linear-gradient(135deg,var(--bg2),var(--surface));
  border-radius:var(--r2) var(--r2) 0 0;
  flex-shrink:0;
}
.chat-avatar{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--pink));display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.chat-header-info{flex:1;}
.chat-header-name{font-family:var(--sans);font-size:.88rem;font-weight:700;color:var(--text);}
.chat-header-status{font-family:var(--mono);font-size:.6rem;color:var(--green);display:flex;align-items:center;gap:.3rem;}
.chat-header-dot{width:5px;height:5px;border-radius:50%;background:var(--green);}
.chat-close{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:.3rem .55rem;cursor:pointer;font-size:.75rem;color:var(--text3);transition:all .15s;font-family:var(--mono);}
.chat-close:hover{border-color:var(--red);color:var(--red);}
.chat-messages{flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:.65rem;}
.chat-messages::-webkit-scrollbar{width:3px;}
.chat-messages::-webkit-scrollbar-thumb{background:var(--surface2);}
.chat-msg{max-width:86%;display:flex;flex-direction:column;gap:.2rem;}
.chat-msg.user{align-self:flex-end;align-items:flex-end;}
.chat-msg.bot{align-self:flex-start;align-items:flex-start;}
.chat-bubble{padding:.6rem .875rem;border-radius:12px;font-size:.82rem;line-height:1.5;}
.chat-msg.bot .chat-bubble{background:var(--bg3);border:1px solid var(--border);color:var(--text2);border-radius:4px 12px 12px 12px;}
.chat-msg.user .chat-bubble{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border-radius:12px 4px 12px 12px;}
.chat-time{font-family:var(--mono);font-size:.55rem;color:var(--text3);}
.chat-typing{display:flex;gap:.3rem;padding:.6rem .875rem;align-items:center;}
.chat-typing span{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:typingDot 1.2s ease-in-out infinite;}
.chat-typing span:nth-child(2){animation-delay:.2s;}
.chat-typing span:nth-child(3){animation-delay:.4s;}
@keyframes typingDot{0%,60%,100%{transform:translateY(0);opacity:.4;}30%{transform:translateY(-5px);opacity:1;}}
.chat-input-area{display:flex;align-items:center;gap:.5rem;padding:.75rem 1rem;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0;}
.chat-input{flex:1;background:var(--bg3);border:1px solid var(--border2);color:var(--text);border-radius:20px;padding:.5rem .875rem;font-family:var(--sans);font-size:.82rem;outline:none;resize:none;max-height:80px;transition:border-color .2s;}
.chat-input:focus{border-color:var(--accent);}
.chat-send{width:34px;height:34px;border-radius:50%;background:var(--accent);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#fff;font-size:.9rem;flex-shrink:0;transition:all .15s;}
.chat-send:hover{background:var(--accent2);transform:scale(1.05);}
.chat-send:disabled{opacity:.4;cursor:not-allowed;transform:none;}
.chat-suggestions{display:flex;gap:.4rem;flex-wrap:wrap;padding:.5rem 1rem;border-top:1px solid var(--border);background:var(--bg2);}
.chat-sug{font-family:var(--mono);font-size:.6rem;padding:.25rem .6rem;border-radius:12px;border:1px solid var(--border2);background:var(--bg3);color:var(--text3);cursor:pointer;transition:all .15s;white-space:nowrap;}
.chat-sug:hover{border-color:var(--accent);color:var(--accent);}
@media (max-width: 768px){
  .chat-panel.expanded{
    right:0;
    bottom:0;
    width:100vw;
    height:100vh;
    border-radius:0;
  }
}
`;

// ── UTILS ──────────────────────────────────────────────────────────────────
const CURRENCY_RATES = {LKR:1,USD:320,EUR:345,GBP:400,AUD:210,SGD:235};
function toLKR(amount,currency){return amount*(CURRENCY_RATES[currency]||1);}
function formatLKR(val){return "LKR "+Math.round(val).toLocaleString();}
function normalizeGpa(val,system){
  const v=parseFloat(val)||0;
  if(system==="GPA (4.0 scale)") return +v.toFixed(2);
  if(system==="GPA (5.0 scale)") return +((v/5)*4).toFixed(2);
  if(system==="UK Class") return ({"First Class":3.7,"Upper Second (2:1)":3.3,"Lower Second (2:2)":3.0,"Third Class":2.5})[val]??3.0;
  if(system==="Percentage"){if(v>=90)return 4.0;if(v>=85)return 3.7;if(v>=80)return 3.5;if(v>=75)return 3.3;if(v>=70)return 3.0;if(v>=65)return 2.7;if(v>=60)return 2.5;return 2.0;}
  return 3.0;
}

// ── CHATBOT ────────────────────────────────────────────────────────────────
const BOT_INTRO = [
  {id:"intro",role:"bot",text:"Hi! 👋 I'm your UniAssist advisor. Ask me anything about studying abroad — universities, requirements, scholarships, or visas!",time:now()}
];

function now(){return new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});}

function createMessage(role,text,metadata){
  return {
    id:`${Date.now()}-${Math.random().toString(36).slice(2,8)}`,
    role,
    text,
    time:now(),
    metadata: metadata || undefined,
  };
}

function parseBotStructuredSections(text){
  if(typeof text !== "string" || !text.trim()){
    return { body: text || "", options: [], links: [] };
  }

  const lines = text.split("\n");
  const bodyLines = [];
  const options = [];
  const links = [];
  let mode = "body";

  for(const rawLine of lines){
    const line = (rawLine || "").trim();
    const lower = line.toLowerCase();

    if(lower === "suggested options:"){
      mode = "options";
      continue;
    }
    if(lower === "relevant links:"){
      mode = "links";
      continue;
    }

    if(mode === "options" && line.startsWith("- ")){
      const value = line.slice(2).trim();
      if(value) options.push(value);
      continue;
    }

    if(mode === "links" && line.startsWith("- ")){
      const value = line.slice(2).trim();
      if(value) links.push(value);
      continue;
    }

    bodyLines.push(rawLine);
  }

  return {
    body: bodyLines.join("\n").trim(),
    options,
    links,
  };
}

function createIntroMessages(){
  return [{...BOT_INTRO[0]}];
}

function getChatStorageKey(user){
  const email = user?.email?.trim().toLowerCase();
  return email ? `uniassist.chat.${email}` : null;
}

function isIntroMessage(message){
  return message?.id === "intro" || (message?.role === "bot" && message?.text === BOT_INTRO[0].text);
}

function dedupeMessages(messages){
  const seen = new Set();
  return messages.filter(message=>{
    const key = message?.id || `${message?.role||""}|${message?.text||""}|${message?.time||""}`;
    if(seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function normalizeMessages(messages){
  if(!Array.isArray(messages)) return createIntroMessages();
  const valid = messages
    .filter(message=>message&&typeof message.text==="string"&&typeof message.role==="string")
    .map(message=>({
      id:message.id || undefined,
      role:message.role,
      text:message.text,
      time:message.time || now(),
      metadata: message.metadata || undefined,
    }));
  if(!valid.length) return createIntroMessages();
  const withIntro = valid.some(isIntroMessage) ? valid : [createIntroMessages()[0],...valid];
  return dedupeMessages(withIntro);
}

function hasConversationContent(messages){
  return normalizeMessages(messages).some(message=>!isIntroMessage(message));
}

function authHeaders(token, includeJsonContentType = false){
  const headers = {};
  if(includeJsonContentType) headers["Content-Type"] = "application/json";
  if(token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function fetchChatHistory(userId,token){
  const response = await fetchApi(`/chat/history?user_id=${encodeURIComponent(userId)}`, {
    headers: authHeaders(token),
  });
  if(!response.ok) throw new Error(`Failed to load chat history (${response.status})`);
  const payload = await response.json();
  return normalizeMessages(payload.messages);
}

async function appendChatHistory(userId,messages,token){
  if(!messages.length) return;
  await fetchApi("/chat/history",{
    method:"POST",
    headers:authHeaders(token, true),
    body:JSON.stringify({user_id:userId,messages}),
  });
}

async function clearChatHistory(userId,token){
  await fetchApi(`/chat/history?user_id=${encodeURIComponent(userId)}`,{
    method:"DELETE",
    headers:authHeaders(token),
  });
}

async function fetchUserState(userId,token){
  const response = await fetchApi(`/user/state?user_id=${encodeURIComponent(userId)}`, {
    headers: authHeaders(token),
  });
  if(!response.ok) throw new Error(`Failed to load user state (${response.status})`);
  const payload = await response.json();
  return payload?.state || {};
}

async function saveUserState(userId,state,token){
  await fetchApi("/user/state", {
    method: "POST",
    headers: authHeaders(token, true),
    body: JSON.stringify({ user_id: userId, state }),
  });
}

async function fetchUserDocuments(token){
  const response = await fetchApi("/documents", {
    headers: authHeaders(token),
  });
  if(!response.ok) throw new Error(`Failed to load documents (${response.status})`);
  const payload = await response.json();
  return Array.isArray(payload?.documents) ? payload.documents : [];
}

async function deleteUserDocument(documentId,token){
  const response = await fetchApi(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if(!response.ok){
    const payload = await response.json().catch(()=>({}));
    throw new Error(payload?.detail || `Delete failed (${response.status})`);
  }
}

async function openUserDocument(documentId,token){
  const response = await fetchApi(`/documents/${encodeURIComponent(documentId)}/content`, {
    headers: authHeaders(token),
  });
  if(!response.ok){
    const payload = await response.json().catch(()=>({}));
    throw new Error(payload?.detail || `Open failed (${response.status})`);
  }
  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  window.open(blobUrl, "_blank", "noopener,noreferrer");
  setTimeout(()=>URL.revokeObjectURL(blobUrl), 60000);
}

async function registerUser({name,email,password,role="student"}){
  const normalizedName = (name || "").trim();
  const normalizedEmail = (email || "").trim().toLowerCase();
  const response = await fetchApi("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: normalizedName, email: normalizedEmail, password, role }),
  });
  const payload = await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(payload?.detail || `Register failed (${response.status})`);
  return payload;
}

async function loginUser({email,password}){
  const normalizedEmail = (email || "").trim().toLowerCase();
  const response = await fetchApi("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: normalizedEmail, password }),
  });
  const payload = await response.json().catch(()=>({}));
  if(!response.ok){
    const detail = payload?.detail || `Login failed (${response.status})`;
    if(String(detail).toLowerCase().includes("invalid email or password")){
      throw new Error("Invalid email or password. If this is a new Colab session, please register first because local users reset between sessions.");
    }
    throw new Error(detail);
  }
  return payload;
}

async function fetchBackendChatReply(message, token, messages, timeoutMs = 12000){
  // Send last 20 turns as conversation_history so Gemini can maintain context
  const conversation_history = (messages || [])
    .filter(m => m.role === "user" || m.role === "bot")
    .slice(-20)
    .map(m => ({ role: m.role === "bot" ? "assistant" : m.role, text: m.text }));
  const response = await fetchApi("/chat/respond", {
    method: "POST",
    headers: authHeaders(token, true),
    body: JSON.stringify({ user_message: message, context: { conversation_history } }),
    timeoutMs,
  });
  if(!response.ok) throw new Error(`Chat backend failed (${response.status})`);
  const payload = await response.json();
  const text = (payload?.response || "").trim();
  if(!text){
    throw new Error("Chat backend returned an empty response");
  }
  return {
    text,
    source: payload?.source || "backend_agent",
    intent: payload?.intent,
    actions: payload?.actions,
    agentData: payload?.agent_data,
    externalFactors: payload?.agent_data?.external_factors || [],
  };
}

const QUICK_REPLIES = [
  "Best unis for CS in UK?",
  "IELTS requirements?",
  "Scholarships for Sri Lankans",
  "Cost of living in Singapore",
];

const FACTOR_QUICK_REPLIES = [
  { label: "Financial", text: "My family budget is limited. Which universities are affordable with scholarship options?" },
  { label: "Eligibility", text: "I have A/L results and IELTS. Am I eligible for UK or Australia CS programs?" },
  { label: "Visa", text: "What visa documents should I prepare early to avoid rejection risk?" },
  { label: "Deadlines", text: "I am late for this intake. Which universities still have open deadlines?" },
  { label: "Trust", text: "Why are these universities recommended for me? Show the reason clearly." },
  { label: "Global Risk", text: "If policies change or travel restrictions return, what backup plan should I use?" },
];

function ChatBot({user}){
  const [open,setOpen]=useState(false);
  const [expanded,setExpanded]=useState(false);
  const [messages,setMessages]=useState(()=>createIntroMessages());
  const [input,setInput]=useState("");
  const [typing,setTyping]=useState(false);
  const [unread,setUnread]=useState(0);
  const messagesEndRef=useRef(null);
  const inputRef=useRef(null);
  const storageKey = getChatStorageKey(user);

  useEffect(()=>{
    let cancelled = false;
    const localMessages = (()=>{
      if(!storageKey) return createIntroMessages();
      try{
        const raw = window.localStorage.getItem(storageKey);
        return raw ? normalizeMessages(JSON.parse(raw)) : createIntroMessages();
      }catch{
        return createIntroMessages();
      }
    })();

    setMessages(localMessages);
    setUnread(0);

    if(!storageKey || !user?.email) return ()=>{cancelled=true;};

    (async()=>{
      try{
        const remoteMessages = await fetchChatHistory(user.email, user?.token);
        if(cancelled) return;
        setMessages(currentMessages=>normalizeMessages([...currentMessages,...remoteMessages]));
        if(hasConversationContent(localMessages) && !hasConversationContent(remoteMessages)){
          await appendChatHistory(user.email, localMessages.filter(message=>!isIntroMessage(message)), user?.token);
        }
      }catch{
        if(cancelled) return;
      }
    })();

    return ()=>{cancelled=true;};
  },[storageKey,user]);

  useEffect(()=>{
    if(!storageKey) return;
    try{
      window.localStorage.setItem(storageKey, JSON.stringify(normalizeMessages(messages)));
    }catch{
      // Ignore storage quota and private mode failures.
    }
  },[messages,storageKey]);

  useEffect(()=>{
    if(open){
      setUnread(0);
      setTimeout(()=>messagesEndRef.current?.scrollIntoView({behavior:"smooth"}),100);
      setTimeout(()=>inputRef.current?.focus(),200);
    }
  },[open,messages]);

  useEffect(()=>{
    if(!open && expanded){
      setExpanded(false);
    }
  },[open,expanded]);

  const sendMessage=async(text)=>{
    const msg=text||input.trim();
    if(!msg)return;
    setInput("");
    const userMessage = createMessage("user",msg);
    setMessages(p=>[...p,userMessage]);
    if(user?.email){
      appendChatHistory(user.email,[userMessage], user?.token).catch(()=>{});
    }
    setTyping(true);
    await new Promise(r=>setTimeout(r,350));
    let replyText = "";
    let replySource = "backend_agent";
    let replyIntent = undefined;
    let replyActions = undefined;
    let replyAgentData = undefined;
    let replyExternalFactors = [];
    try{
      const backendReply = await fetchBackendChatReply(msg, user?.token, messages);
      replyText = (backendReply.text || "").trim();
      replySource = backendReply.source || "backend_agent";
      replyIntent = backendReply.intent;
      replyActions = backendReply.actions;
      replyAgentData = backendReply.agentData;
      replyExternalFactors = backendReply.externalFactors;
    }catch(error){
      const message = error instanceof Error ? error.message : "Live chat unavailable";
      replyText = `Live backend chat is unavailable right now. ${message}. Please check the API server and try again.`;
      replySource = "backend_error";
      replyIntent = "error";
      replyActions = ["Verify API server is running", "Retry your message"];
    }
    setTyping(false);
    const botMessage = createMessage("bot",replyText,{
      source: replySource,
      intent: replyIntent,
      actions: replyActions,
      agentData: replyAgentData,
      externalFactors: replyExternalFactors,
    });
    setMessages(p=>[...p,botMessage]);
    if(user?.email){
      appendChatHistory(user.email,[botMessage], user?.token).catch(()=>{});
    }
    if(!open) setUnread(n=>n+1);
  };

  const handleClearChat=async()=>{
    setMessages(createIntroMessages());
    setUnread(0);
    if(storageKey){
      try{
        window.localStorage.removeItem(storageKey);
      }catch{
        // Ignore storage failures during clear.
      }
    }
    if(user?.email){
      try{
        await clearChatHistory(user.email, user?.token);
      }catch{
        // Keep local clear even if backend clear fails.
      }
    }
  };

  const handleKey=(e)=>{
    if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage();}
  };

  return(
    <>
      <button className="chat-fab" onClick={()=>setOpen(p=>!p)} title="Chat with AI Advisor">
        {open ? "✕" : "💬"}
        {!open && unread>0 && <div className="chat-fab-badge">{unread}</div>}
      </button>
      <div className={`chat-panel${open?" open":""}${expanded?" expanded":""}`}>
        <div className="chat-header">
          <div className="chat-avatar">🎓</div>
          <div className="chat-header-info">
            <div className="chat-header-name">UniAssist AI Advisor</div>
            <div className="chat-header-status"><div className="chat-header-dot"/>Online · Always ready</div>
          </div>
          <button className="chat-close" onClick={handleClearChat} type="button">Clear</button>
          <button
            className="chat-close"
            onClick={()=>setExpanded(p=>!p)}
            type="button"
            aria-label={expanded?"Collapse chat":"Expand chat"}
            title={expanded?"Collapse chat":"Expand chat"}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
          <button className="chat-close" onClick={()=>setOpen(false)}>✕</button>
        </div>
        <div className="chat-messages">
          {messages.map((m,i)=>(
            <div key={m.id||i} className={`chat-msg ${m.role}`}>
              <div className="chat-bubble">
                {(()=>{
                  const parsed = m.role === "bot" ? parseBotStructuredSections(m.text) : null;
                  const displayText = parsed ? (parsed.body || m.text) : m.text;
                  return <div>{displayText}</div>;
                })()}
                {m.role === "bot" && Array.isArray(m.metadata?.externalFactors) && m.metadata.externalFactors.length > 0 && (
                  <div style={{marginTop:".55rem", display:"flex", gap:".35rem", flexWrap:"wrap"}}>
                    {m.metadata.externalFactors.map(f => (
                      <span
                        key={f.id || f.label}
                        style={{fontSize:".72rem", padding:".15rem .45rem", borderRadius:"999px", background:"rgba(255,255,255,.12)", border:"1px solid rgba(255,255,255,.18)"}}
                      >
                        {f.label}
                      </span>
                    ))}
                  </div>
                )}
                {m.role === "bot" && (()=>{
                  const parsed = parseBotStructuredSections(m.text);
                  if(!parsed.options.length) return null;
                  return (
                    <div style={{marginTop:".55rem", display:"flex", gap:".35rem", flexWrap:"wrap"}}>
                      {parsed.options.slice(0,3).map(option => (
                        <span
                          key={option}
                          style={{fontSize:".72rem", padding:".15rem .45rem", borderRadius:"999px", background:"rgba(79,124,255,0.12)", border:"1px solid rgba(79,124,255,0.35)"}}
                        >
                          {option}
                        </span>
                      ))}
                    </div>
                  );
                })()}
                {m.role === "bot" && (()=>{
                  const parsed = parseBotStructuredSections(m.text);
                  if(!parsed.links.length) return null;
                  return (
                    <div style={{marginTop:".55rem", display:"flex", gap:".35rem", flexWrap:"wrap"}}>
                      {parsed.links.slice(0,3).map(url => (
                        <a
                          key={url}
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          style={{fontSize:".72rem", padding:".15rem .45rem", borderRadius:"999px", textDecoration:"none", color:"#dbe7ff", background:"rgba(44,159,176,0.18)", border:"1px solid rgba(44,159,176,0.42)"}}
                        >
                          Open source
                        </a>
                      ))}
                    </div>
                  );
                })()}
                {m.role === "bot" && Array.isArray(m.metadata?.actions) && m.metadata.actions.length > 0 && (
                  <div style={{marginTop:".5rem", fontSize:".78rem", opacity:.9}}>
                    Next: {m.metadata.actions.slice(0, 3).join(" • ")}
                  </div>
                )}
              </div>
              <div className="chat-time">{m.time}</div>
            </div>
          ))}
          {typing&&(
            <div className="chat-msg bot">
              <div className="chat-bubble" style={{padding:".5rem .875rem"}}>
                <div className="chat-typing"><span/><span/><span/></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef}/>
        </div>
        <div className="chat-suggestions">
          {QUICK_REPLIES.map(q=>(
            <button key={q} className="chat-sug" onClick={()=>sendMessage(q)}>{q}</button>
          ))}
        </div>
        <div className="chat-suggestions" style={{borderTop:"1px solid rgba(255,255,255,.08)", paddingTop:".5rem"}}>
          {FACTOR_QUICK_REPLIES.map(item=>(
            <button key={item.label} className="chat-sug" onClick={()=>sendMessage(item.text)} title={item.text}>
              {item.label}
            </button>
          ))}
        </div>
        <div className="chat-input-area">
          <textarea ref={inputRef} className="chat-input" value={input} onChange={e=>setInput(e.target.value)} onKeyDown={handleKey} placeholder="Ask anything about studying abroad…" rows={1}/>
          <button className="chat-send" onClick={()=>sendMessage()} disabled={!input.trim()||typing}>➤</button>
        </div>
      </div>
    </>
  );
}

// ── OCR BACKEND ─────────────────────────────────────────────────────────────
const API_DOC_TYPE_MAP = {
  "A-Level Results": "alevel",
  "Bachelor's Degree": "bachelor",
  "Master's Degree": "master",
  Diploma: "diploma",
  "IELTS Certificate": "ielts",
  "TOEFL Certificate": "toefl",
  "PTE Certificate": "pte",
  Passport: "passport",
  "Financial Statement": "financial",
};

const UPLOAD_DOC_TYPE_OPTIONS = [
  { value: "auto", label: "Auto identify" },
  ...Object.keys(API_DOC_TYPE_MAP).map((label) => ({ value: API_DOC_TYPE_MAP[label], label })),
];

async function extractDocumentWithAI(file, docType, onProgress, token) {
  onProgress("Reading file...", 10);
  const formData = new FormData();
  formData.append("file", file, file.name || "document");
  formData.append("doc_type", docType || "auto");
  onProgress("Preprocessing image with OpenCV...", 25);
  await new Promise(r => setTimeout(r, 180));
  onProgress("Running OCR text extraction...", 45);
  await new Promise(r => setTimeout(r, 120));
  onProgress("Classifying document with ML model...", 60);
  let response;
  try {
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    response = await fetchApi("/ocr", { method: "POST", headers, body: formData });
  } catch (netErr) {
    throw new Error(netErr?.message || "Cannot reach OCR server. Start the API locally on port 8000 or update VITE_API_URL.");
  }
  if (!response.ok) {
    const errBody = await response.json().catch(() => ({}));
    throw new Error(errBody?.detail || `Server error ${response.status}`);
  }
  onProgress("Extracting structured fields...", 75);
  const result = await response.json();
  if (!result.success) throw new Error((result.errors || []).join(" ") || "Extraction failed");
  onProgress("Scoring confidence...", 90);
  await new Promise(r => setTimeout(r, 80));
  onProgress("Extraction complete!", 100);
  return {
    success:true,
    data:result.data,
    confidence:result.confidence??0.75,
    rawPreview:result.raw_text_preview||"",
    message:result.message||`OCR — ${Math.round((result.confidence??0.75)*100)}% confidence`,
    ocrEngine:result.ocr_engine||"auto",
    warnings:result.warnings||[],
    requested_doc_type:docType||"auto",
    classification_method:result.classification_method,
  };
}

// ── ELIGIBILITY ENGINE ──────────────────────────────────────────────────────
function assessEligibility(profile, docPayload) {
  const asArray = (value)=>Array.isArray(value)?value.filter(Boolean):[];
  const toTierScore = (tier)=>({foundation:1,average:2,good:3,top:4}[tier]||1);
  const fromTierScore = (score)=>({1:"foundation",2:"average",3:"good",4:"top"}[Math.max(1,Math.min(4,Math.round(score)))]||"foundation");

  const docsByType = (()=>{
    if(docPayload?.documents && typeof docPayload.documents === "object"){
      return docPayload.documents;
    }
    if(docPayload?.document_type){
      return {[docPayload.document_type]:docPayload};
    }
    return {};
  })();

  const docs = Object.values(docsByType).filter((d)=>d && typeof d === "object");
  if(!docs.length){
    return {
      eligible:false,
      grade_point:0,
      eligible_countries:[],
      recommended_programs:[],
      eligibility_tier:"foundation",
      notes:["No documents available. Upload academic, English, passport, and financial documents."],
      checks:{academic:false,english:false,passport:false,financial:false},
      documents_considered:[],
    };
  }

  const findDoc = (...types)=>docs.find((d)=>types.includes(d.document_type));
  const academicDoc = findDoc("Master's Degree","Bachelor's Degree","A-Level Results","Diploma");
  const ieltsDoc = findDoc("IELTS Certificate");
  const toeflDoc = findDoc("TOEFL Certificate");
  const pteDoc = findDoc("PTE Certificate");
  const passportDoc = findDoc("Passport");
  const financialDoc = findDoc("Financial Statement");

  let academicResult = {eligible:false,grade_point:0,eligibility_tier:"foundation",eligible_countries:["UK","Singapore","Australia"],recommended_programs:[],notes:["Academic document missing."]};
  if(academicDoc){
    if(academicDoc.document_type==="A-Level Results"){
      const GV={A:4,B:3,C:2,S:1,F:0};
      const subjectMap = academicDoc.subjects && !Array.isArray(academicDoc.subjects) ? academicDoc.subjects : {};
      const vals=Object.values(subjectMap).map(g=>GV[String(g).trim()]??0).filter((v)=>Number.isFinite(v));
      const avg=vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 2.0;
      const gp=+avg.toFixed(2);
      academicResult = avg>=3.7
        ? {eligible:true,grade_point:gp,eligibility_tier:"top",eligible_countries:["UK","Singapore","Australia"],recommended_programs:["Engineering","Computer Science","Medicine","Business"],notes:["Excellent A/L grades — strong academic eligibility."]}
        : avg>=3.3
          ? {eligible:true,grade_point:gp,eligibility_tier:"good",eligible_countries:["UK","Singapore","Australia"],recommended_programs:["Engineering","Business","Science"],notes:["Strong A/L profile for direct entry."]}
          : avg>=3.0
            ? {eligible:true,grade_point:gp,eligibility_tier:"average",eligible_countries:["UK","Australia"],recommended_programs:["Business","IT","Science"],notes:["Moderate A/L profile."]}
            : {eligible:false,grade_point:gp,eligibility_tier:"foundation",eligible_countries:["UK","Australia"],recommended_programs:["Foundation Program","Pathway"],notes:["A/L grades indicate pathway/foundation route."]};
    }else{
      const gpa=parseFloat(academicDoc.gpa_normalized??academicDoc.gpa_value??academicDoc.gpa??0) || 0;
      const isMasters=academicDoc.document_type==="Master's Degree";
      academicResult = gpa>=3.5
        ? {eligible:true,grade_point:gpa,eligibility_tier:"top",eligible_countries:["UK","Singapore","Australia"],recommended_programs:isMasters?["PhD","Research Programs","Postdoc"]:["Master's","MBA","PhD"],notes:[`Excellent ${academicDoc.document_type} result.`]}
        : gpa>=3.0
          ? {eligible:true,grade_point:gpa,eligibility_tier:"good",eligible_countries:["UK","Singapore","Australia"],recommended_programs:isMasters?["PhD","Second Master's"]:["Master's","MBA"],notes:[`Good ${academicDoc.document_type} result.`]}
          : gpa>=2.5
            ? {eligible:true,grade_point:gpa,eligibility_tier:"average",eligible_countries:["Australia","UK"],recommended_programs:["Master's","Graduate Diploma"],notes:["Academic profile is acceptable for selected programs."]}
            : {eligible:false,grade_point:gpa,eligibility_tier:"foundation",eligible_countries:["Australia"],recommended_programs:["Graduate Diploma","Foundation"],notes:["Academic profile needs pathway/bridging route."]};
    }
  }

  let englishResult = {eligible:false,grade_point:0,eligibility_tier:"foundation",notes:["English certificate missing (IELTS/TOEFL/PTE)."]};
  if(ieltsDoc){
    const o=parseFloat(ieltsDoc.overall||0) || 0;
    const tier=o>=7.5?"top":o>=6.5?"good":o>=6.0?"average":"foundation";
    englishResult={eligible:o>=5.5,grade_point:o/9*4,eligibility_tier:tier,notes:[`IELTS Overall: ${o}`]};
  }else if(toeflDoc){
    const t=parseInt(toeflDoc.total||0,10) || 0;
    const tier=t>=100?"top":t>=90?"good":t>=80?"average":"foundation";
    englishResult={eligible:t>=72,grade_point:t/120*4,eligibility_tier:tier,notes:[`TOEFL iBT: ${t}`]};
  }else if(pteDoc){
    const p=parseInt(pteDoc.overall||0,10) || 0;
    const tier=p>=79?"top":p>=65?"good":p>=50?"average":"foundation";
    englishResult={eligible:p>=50,grade_point:p/90*4,eligibility_tier:tier,notes:[`PTE Academic: ${p}`]};
  }else if(docPayload?.english_proficiency?.ielts?.overall){
    const o=parseFloat(docPayload.english_proficiency.ielts.overall||0) || 0;
    const tier=o>=7.5?"top":o>=6.5?"good":o>=6.0?"average":"foundation";
    englishResult={eligible:o>=5.5,grade_point:o/9*4,eligibility_tier:tier,notes:[`IELTS (manual): ${o}`]};
  }

  const financialProfile=profile.financial||{};
  const profileBudgetLKR=toLKR(parseFloat(financialProfile.total_budget)||0,financialProfile.budget_currency||"LKR");
  let financialLKR=profileBudgetLKR;
  if(financialDoc){
    const cb=parseFloat(String(financialDoc.closing_balance||"0").replace(/[^0-9.]/g,""))||0;
    financialLKR=toLKR(cb,financialDoc.currency||"LKR");
  }
  const financialTier=financialLKR>=10000000?"top":financialLKR>=5000000?"good":financialLKR>0?"average":"foundation";
  const financialOk=financialLKR>0;

  const passportOk=!!passportDoc;
  const checks={
    academic:!!academicDoc && academicResult.eligible,
    english:englishResult.eligible,
    passport:passportOk,
    financial:financialOk,
  };

  const combinedTierScore=(toTierScore(academicResult.eligibility_tier)*0.55)
    +(toTierScore(englishResult.eligibility_tier)*0.25)
    +(toTierScore(financialTier)*0.20);
  const combinedGradePoint=((academicResult.grade_point||0)*0.8)+((englishResult.grade_point||0)*0.2);

  const allCountries=["UK","Singapore","Australia"];
  const countries=academicResult.eligible_countries?.length?academicResult.eligible_countries:allCountries;
  const eligible = checks.academic && checks.english && checks.passport && checks.financial;

  const notes=[
    ...asArray(academicResult.notes),
    ...asArray(englishResult.notes),
    financialOk ? `Financial capacity assessed: ${formatLKR(financialLKR)}.` : "Financial proof missing. Upload bank statement or set profile budget.",
    passportOk ? "Passport verified." : "Passport missing. Upload passport for full application readiness.",
  ];

  return {
    eligible,
    grade_point:+combinedGradePoint.toFixed(2),
    eligible_countries:countries,
    recommended_programs:academicResult.recommended_programs||[],
    eligibility_tier:fromTierScore(combinedTierScore),
    notes,
    checks,
    financial_ok:financialOk,
    documents_considered:docs.map((d)=>d.document_type).filter(Boolean),
  };
}

const UNIVERSITIES_API = "/universities";

const TUITION_RATES = { UK: 400, Singapore: 235, Australia: 210 };

function _mapUniversity(u, country, budgetLKR) {
  const rate = TUITION_RATES[u.country] || 400;
  const tuitionUSD = u.tuition?.undergraduate_intl_gbp
    || u.tuition?.undergraduate_intl_sgd
    || u.tuition?.undergraduate_intl_aud
    || 0;
  const tuitionLKR = tuitionUSD * rate;
  const qs = u.rankings?.qs_world || 999;
  const currencyLabel = u.country === "UK" ? "GBP" : u.country === "Singapore" ? "SGD" : "AUD";
  return {
    name: u.name,
    location: u.country,
    qs,
    programs: u.programs || [],
    tuitionLKR,
    minGpa: u.acceptance_criteria?.min_grade_point || 0,
    ielts: u.acceptance_criteria?.ielts_min || 0,
    website: u.website || "#",
    affordable: budgetLKR <= 0 || budgetLKR >= tuitionLKR,
    tuitionDisplay: `${currencyLabel} ${Math.round(tuitionUSD).toLocaleString()} / yr`,
  };
}

async function fetchUniversities(country, minGpa, program, budgetLKR, token) {
  const url = country ? `${UNIVERSITIES_API}?country=${encodeURIComponent(country)}` : UNIVERSITIES_API;
  const res = await fetchApi(url, token);
  if (!res.ok) throw new Error("Failed to load universities.");
  const { universities } = await res.json();
  return universities
    .map(u => _mapUniversity(u, country, budgetLKR))
    .filter(u => {
      if (minGpa && u.minGpa > minGpa + 0.05) return false;
      if (program && program !== "Other" && !u.programs.includes(program)) return false;
      return true;
    })
    .sort((a, b) => a.qs - b.qs);
}

async function submitApplication(payload, token) {
  const res = await fetchApi("/applications", token, { method: "POST", body: JSON.stringify(payload) });
  if (res.status === 409) throw new Error("Already applied to this university and program.");
  if (!res.ok) { const t = await res.text().catch(() => ""); throw new Error(t || "Submission failed."); }
  return res.json();
}

async function fetchApplications(token) {
  const res = await fetchApi("/applications", token);
  if (!res.ok) throw new Error("Failed to load applications.");
  const { applications } = await res.json();
  return applications;
}

async function updateApplicationStatus(appId, status, advisorNotes, token) {
  const res = await fetchApi(`/applications/${appId}/status`, token, {
    method: "PATCH",
    body: JSON.stringify({ status, advisor_notes: advisorNotes || null }),
  });
  if (!res.ok) { const t = await res.text().catch(() => ""); throw new Error(t || "Update failed."); }
  return res.json();
}

async function withdrawApplication(appId, token) {
  const res = await fetchApi(`/applications/${appId}`, token, { method: "DELETE" });
  if (!res.ok) { const t = await res.text().catch(() => ""); throw new Error(t || "Withdraw failed."); }
  return res.json();
}

// ── SHARED UI ──────────────────────────────────────────────────────────────
function Alert({type="info",children}){
  const cls={info:"alert-info",warn:"alert-warn",error:"alert-error",ok:"alert-ok",ai:"alert-ai"}[type]||"alert-info";
  const icons={info:"ℹ",warn:"⚠",error:"✕",ok:"✓",ai:"✦"};
  return <div className={`alert ${cls}`}><span>{icons[type]}</span><span>{children}</span></div>;
}
function DataGrid({pairs}){
  const valid=pairs.filter(([,v])=>v&&String(v)!=="N/A"&&String(v)!=="undefined"&&String(v)!=="null"&&v!==null);
  return <div className="data-grid">{valid.map(([l,v])=>(<div className="data-cell" key={l}><div className="data-label">{l}</div><div className="data-val">{String(v)}</div></div>))}</div>;
}
function ScoreSections({sections,maxVal=9}){
  return <div className="score-section-grid">{sections.map(([label,val,max])=>{const v=parseFloat(val)||0,mx=max||maxVal,pct=Math.min((v/mx)*100,100);return(<div className="score-section-cell" key={label}><div className="score-section-label">{label}</div><div className="score-section-val">{val||"—"}</div><div className="score-section-bar"><div className="score-section-fill" style={{width:`${pct}%`}} /></div></div>);})}</div>;
}

// ── EXTRACTED DATA DISPLAY ──────────────────────────────────────────────────
// CHANGE v6: A-Level Results section NO LONGER shows english_proficiency.
//            English proficiency must be uploaded as a separate document.
function ExtractedDisplay({data,confidence,ocrEngine}){
  const [showRaw,setShowRaw]=useState(false);
  const dt=data.document_type;
  const engineLabel=(ocrEngine||"ocr").replace(/^./,c=>c.toUpperCase());
  const pct=Math.round((confidence||0)*100);
  const confCls=pct>=80?"conf-high":pct>=60?"conf-mid":"conf-low";
  const renderContent=()=>{
    if(dt==="A-Level Results"){
      // ✕ english_proficiency intentionally excluded — it is a separate document
      const SPECIAL=new Set(["general english","common general test","cgt"]);
      const all=data.subjects||{};
      const main=Object.entries(all).filter(([s])=>!SPECIAL.has(s.toLowerCase()));
      const spec=Object.entries(all).filter(([s])=>SPECIAL.has(s.toLowerCase()));
      return(
        <>
          <DataGrid pairs={[
            ["Full Name",data.full_name],
            ["Index Number",data.index_number],
            ["Year",data.year],
            ["Stream",data.subject_stream],
            ["Z-Score",data.z_score],
            ["District Rank",data.district_rank],
            ["Island Rank",data.island_rank],
          ]} />
          {main.length>0&&<>
            <div className="slabel">Main Subjects</div>
            <table className="subjects-table">
              <thead><tr><th>Subject</th><th>Grade</th></tr></thead>
              <tbody>{main.map(([s,g])=>(<tr key={s}><td>{s}</td><td><span className={`grade-chip gchip-${g}`}>{g}</span></td></tr>))}</tbody>
            </table>
          </>}
          {spec.length>0&&<>
            <div className="slabel">General English</div>
            <table className="subjects-table">
              <tbody>{spec.map(([s,g])=>(<tr key={s}><td>{s}</td><td><span className={`grade-chip gchip-${g}`}>{g}</span></td></tr>))}</tbody>
            </table>
          </>}
          {!main.length&&<Alert type="warn">No subject grades extracted. Please verify manually.</Alert>}
          {/* NOTE: No english_proficiency block here. Upload IELTS/TOEFL/PTE as a separate document. */}
        </>
      );
    }
    if(dt==="Bachelor's Degree"||dt==="Master's Degree"){const isMasters=dt==="Master's Degree";return<><DataGrid pairs={[["Student Name",data.student_name],["Student ID",data.student_id],["University",data.university],["Faculty",data.faculty],["Degree Title",data.degree_title],["Degree Class",data.degree_class],["GPA",data.gpa_value?`${data.gpa_value} / ${data.gpa_scale||"4.0"}`:null],["Percentage",data.percentage],["Graduation",data.graduation_date]]} />{isMasters&&<DataGrid pairs={[["Thesis",data.thesis_title],["Supervisor",data.supervisor]]} />}</>;}
    if(dt==="Diploma")return<DataGrid pairs={[["Student Name",data.student_name],["Institution",data.institution],["Program",data.program],["Grade",data.grade],["Completion",data.completion_year]]} />;
    if(dt==="IELTS Certificate")return<><DataGrid pairs={[["Candidate",data.candidate_name],["Overall Band",`${data.overall} / 9.0`],["Test Date",data.test_date],["TRF Number",data.trf_number],["Test Centre",data.test_centre],["DOB",data.date_of_birth],["Nationality",data.nationality]]} /><div className="slabel">Section Scores</div><ScoreSections sections={[["Listening",data.listening,9],["Reading",data.reading,9],["Writing",data.writing,9],["Speaking",data.speaking,9]]} maxVal={9} /></>;
    if(dt==="TOEFL Certificate")return<><DataGrid pairs={[["Candidate",data.candidate_name],["Total Score",`${data.total} / 120`],["Test Date",data.test_date],["Reg. Number",data.registration_number]]} /><div className="slabel">Section Scores</div><ScoreSections sections={[["Reading",data.reading,30],["Listening",data.listening,30],["Speaking",data.speaking,30],["Writing",data.writing,30]]} maxVal={30} /></>;
      if(dt==="PTE Certificate")return<><DataGrid pairs={[["Candidate",data.candidate_name],["Overall Score",`${data.overall} / 90`],["Test Date",data.test_date]]} /><div className="slabel">Section Scores</div><ScoreSections sections={[["Listening",data.listening,90],["Reading",data.reading,90],["Writing",data.writing,90],["Speaking",data.speaking,90]]} maxVal={90} /></>;
    if(dt==="Passport")return<DataGrid pairs={[["Surname",data.surname],["Given Names",data.given_names],["Passport No.",data.passport_number],["Nationality",data.nationality],["Date of Birth",data.date_of_birth],["Sex",data.sex],["Place of Birth",data.place_of_birth],["Issue Date",data.issue_date],["Expiry",data.expiry_date],["Issuing Authority",data.issuing_authority]]} />;
    if(dt==="Financial Statement")return<><DataGrid pairs={[["Account Holder",data.account_holder],["Bank",data.bank_name],["Account No.",data.account_number],["Currency",data.currency],["Opening Balance",data.opening_balance],["Closing Balance",data.closing_balance],["Period",data.statement_period]]} />{data.closing_balance&&<div style={{marginTop:".75rem",padding:".75rem 1rem",background:"var(--green-dim)",border:"1px solid #3ecf8e22",borderRadius:"var(--r)",fontFamily:"var(--mono)",fontSize:".8rem",color:"var(--green)"}}>💰 Closing Balance: <strong>{data.closing_balance} {data.currency||"LKR"}</strong></div>}</>;
    const skip=new Set(["document_type","confidence"]);
    const pairs=Object.entries(data).filter(([k,v])=>!skip.has(k)&&v&&String(v)!=="N/A"&&typeof v!=="object").map(([k,v])=>[k.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase()),String(v)]);
    return<DataGrid pairs={pairs} />;
  };
  return(
    <div>
      <div className="ai-status-bar">
        <span className="ai-badge">🔍 {engineLabel}</span>
        <span className="ai-status-text">Document extracted via {engineLabel} + ML · {dt}</span>
        <span className={`ai-conf-badge ${confCls}`}>◆ {pct}% confidence</span>
      </div>
      {renderContent()}
      {data._rawPreview&&<div style={{marginTop:"1rem"}}><button className="btn btn-ghost btn-sm" onClick={()=>setShowRaw(p=>!p)}>{showRaw?"▲ Hide":"▼ Show"} Raw OCR Text</button>{showRaw&&<div className="raw-text-box"><pre>{data._rawPreview}</pre></div>}</div>}
    </div>
  );
}

// ── IELTS MANUAL ────────────────────────────────────────────────────────────
const BAND_LABELS=["overall","listening","reading","writing","speaking"];
function bandColor(v){if(v>=7.5)return "var(--accent2)";if(v>=6.5)return "var(--green)";if(v>=6.0)return "var(--teal)";if(v>=5.0)return "var(--amber)";return "var(--red)";}
function bandStatus(v){if(v>=8.5)return "Expert";if(v>=7.5)return "Very Good";if(v>=6.5)return "Competent";if(v>=6.0)return "Acceptable";if(v>=5.5)return "Modest";return "Below Req.";}

function IELTSManualInput({value={},onChange}){
  const init={overall:"",listening:"",reading:"",writing:"",speaking:"",test_date:"",trf_number:"",test_centre:"",...value};
  const [scores,setScores]=useState(init);
  const update=(k,v)=>{const next={...scores,[k]:v};setScores(next);onChange?.(next);};
  const VALID_BANDS=new Set([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9]);
  const valid=(k)=>{const v=parseFloat(scores[k]);return !isNaN(v)&&VALID_BANDS.has(v);};
  return(
    <div>
      <div className="band-grid">{BAND_LABELS.map(k=>{const v=parseFloat(scores[k])||0,isOk=valid(k);return(<div className="band-cell" key={k}><div className="band-label">{k.charAt(0).toUpperCase()+k.slice(1)}</div><input className="band-input" type="number" min={0} max={9} step={0.5} value={scores[k]} placeholder="0.0" onChange={e=>update(k,e.target.value)} /><div className="band-bar"><div className="band-fill" style={{width:`${(v/9)*100}%`,background:isOk?bandColor(v):"var(--border2)"}} /></div>{scores[k]!==""&&<div className="band-status" style={{color:isOk?bandColor(v):"var(--red)"}}>{isOk?bandStatus(v):"Invalid"}</div>}</div>);})}</div>
      <Alert type="info">Valid IELTS bands: 0.0–9.0 in 0.5 steps.</Alert>
      <div className="fgrid" style={{marginTop:"1rem"}}>
        <div className="field"><label className="flabel">Test Date</label><input type="date" value={scores.test_date} onChange={e=>update("test_date",e.target.value)} /></div>
        <div className="field"><label className="flabel">TRF Number</label><input value={scores.trf_number} onChange={e=>update("trf_number",e.target.value)} placeholder="e.g. 20LK000042TEST01" /></div>
        <div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Test Centre</label><input value={scores.test_centre} onChange={e=>update("test_centre",e.target.value)} placeholder="e.g. British Council Colombo" /></div>
      </div>
    </div>
  );
}

// ── ENGLISH SECTION ─────────────────────────────────────────────────────────
function EnglishSection({value={},onChange}){
  const [hasIelts,setHasIelts]=useState(!!value.ielts);
  const [ieltsMode,setIeltsMode]=useState("manual");
  const [ieltsSlider,setIeltsSlider]=useState(value.ielts?.overall??6.5);
  const [ieltsManual,setIeltsManual]=useState(value.ielts||{});
  const [hasToefl,setHasToefl]=useState(!!value.toefl);
  const [toefl,setToefl]=useState(value.toefl??90);
  const [hasPte,setHasPte]=useState(!!value.pte);
  const [pte,setPte]=useState(value.pte??58);
  const build=()=>({toefl:hasToefl?toefl:null,pte:hasPte?pte:null});
  const ic=(v)=>v>=7.5?"var(--accent2)":v>=6.5?"var(--green)":v>=6.0?"var(--teal)":"var(--red)";
  const is=(v)=>v>=7.5?"Excellent":v>=6.5?"Good":v>=6.0?"Acceptable":"Below Req.";
  const tc=toefl>=100?"var(--accent2)":toefl>=90?"var(--green)":toefl>=80?"var(--teal)":"var(--red)";
  const ts=toefl>=100?"Excellent":toefl>=90?"Good":toefl>=80?"Acceptable":"Below Req.";
  const pc=pte>=79?"var(--accent2)":pte>=65?"var(--green)":pte>=50?"var(--teal)":"var(--red)";
  const ps=pte>=79?"Excellent":pte>=65?"Good":pte>=50?"Acceptable":"Below Req.";
  return(
    <div>
      <div className="slabel">English Proficiency</div>
      <div className="eng-panel">
        <div style={{display:"flex",alignItems:"center",gap:".75rem"}}><label className="cbox-row" style={{flex:1}}><input type="checkbox" checked={hasIelts} onChange={e=>{setHasIelts(e.target.checked);onChange?.({ielts:e.target.checked?{overall:ieltsMode==="slider"?ieltsSlider:parseFloat(ieltsManual.overall||0)}:null,...build()});}}/><span>IELTS Academic score</span></label><span className="eng-badge eb-ielts">IELTS</span></div>
        {hasIelts&&<div style={{marginTop:".875rem"}}><div style={{display:"flex",gap:".5rem",marginBottom:"1rem"}}><button className={`btn btn-sm ${ieltsMode==="manual"?"btn-primary":"btn-ghost"}`} onClick={()=>setIeltsMode("manual")}>✏️ Type Scores</button><button className={`btn btn-sm ${ieltsMode==="slider"?"btn-primary":"btn-ghost"}`} onClick={()=>setIeltsMode("slider")}>🎚️ Slider</button></div>{ieltsMode==="manual"?(<IELTSManualInput value={ieltsManual} onChange={m=>{setIeltsManual(m);const o=parseFloat(m.overall||0);onChange?.({...build(),ielts:{overall:o,listening:parseFloat(m.listening||0),reading:parseFloat(m.reading||0),writing:parseFloat(m.writing||0),speaking:parseFloat(m.speaking||0),test_date:m.test_date||"",trf_number:m.trf_number||"",test_centre:m.test_centre||""}});}} />):(<div><div className="score-row"><span>Overall Band</span><span className="score-val">{ieltsSlider.toFixed(1)}</span></div><div className="track"><div className="fill fill-ielts" style={{width:`${(ieltsSlider/9)*100}%`}} /></div><div style={{fontFamily:"var(--mono)",fontSize:".7rem",color:ic(ieltsSlider),marginBottom:".75rem"}}>▸ {is(ieltsSlider)}</div><input type="range" min={0} max={9} step={0.5} value={ieltsSlider} onChange={e=>{const v=+e.target.value;setIeltsSlider(v);onChange?.({ielts:{overall:v},...build()});}}/></div>)}</div>}
      </div>
      <div className="eng-panel">
        <div style={{display:"flex",alignItems:"center",gap:".75rem"}}><label className="cbox-row" style={{flex:1}}><input type="checkbox" checked={hasToefl} onChange={e=>{setHasToefl(e.target.checked);onChange?.({ielts:hasIelts?{overall:parseFloat(ieltsManual.overall||ieltsSlider||0)}:null,toefl:e.target.checked?toefl:null,pte:hasPte?pte:null});}}/><span>TOEFL iBT score</span></label><span className="eng-badge eb-toefl">TOEFL</span></div>
        {hasToefl&&<div style={{marginTop:"1rem"}}><div className="score-row"><span>Total</span><span className="score-val">{toefl} / 120</span></div><div className="track"><div className="fill fill-toefl" style={{width:`${(toefl/120)*100}%`}} /></div><div style={{fontFamily:"var(--mono)",fontSize:".7rem",color:tc,marginBottom:".75rem"}}>▸ {ts}</div><input type="range" min={0} max={120} step={1} value={toefl} onChange={e=>{const v=+e.target.value;setToefl(v);onChange?.({ielts:hasIelts?{overall:parseFloat(ieltsManual.overall||ieltsSlider||0)}:null,toefl:v,pte:hasPte?pte:null});}}/></div>}
      </div>
      <div className="eng-panel">
        <div style={{display:"flex",alignItems:"center",gap:".75rem"}}><label className="cbox-row" style={{flex:1}}><input type="checkbox" checked={hasPte} onChange={e=>{setHasPte(e.target.checked);onChange?.({ielts:hasIelts?{overall:parseFloat(ieltsManual.overall||ieltsSlider||0)}:null,toefl:hasToefl?toefl:null,pte:e.target.checked?pte:null});}}/><span>PTE Academic score</span></label><span className="eng-badge eb-pte">PTE</span></div>
        {hasPte&&<div style={{marginTop:"1rem"}}><div className="score-row"><span>Overall</span><span className="score-val">{pte} / 90</span></div><div className="track"><div className="fill fill-pte" style={{width:`${(pte/90)*100}%`}} /></div><div style={{fontFamily:"var(--mono)",fontSize:".7rem",color:pc,marginBottom:".75rem"}}>▸ {ps}</div><input type="range" min={10} max={90} step={1} value={pte} onChange={e=>{const v=+e.target.value;setPte(v);onChange?.({ielts:hasIelts?{overall:parseFloat(ieltsManual.overall||ieltsSlider||0)}:null,toefl:hasToefl?toefl:null,pte:v});}}/></div>}
      </div>
      {!hasIelts&&!hasToefl&&!hasPte&&<Alert type="warn">No English test selected. Most universities require at least one score.</Alert>}
    </div>
  );
}

// ── FINANCIAL SECTION ───────────────────────────────────────────────────────
function FinancialSection({value={},onChange}){
  const [currency,setCurrency]=useState(value.budget_currency||"LKR");
  const [totalBudget,setTotalBudget]=useState(value.total_budget||"");
  const [annualTuition,setAnnualTuition]=useState(value.annual_tuition_budget||"");
  const [livingCost,setLivingCost]=useState(value.annual_living_budget||"");
  const [fundingSource,setFundingSource]=useState(value.funding_source||"");
  const [scholarships,setScholarships]=useState(value.scholarships_interest||[]);
  const SCHOLARSHIPS=[{id:"commonwealth",label:"Commonwealth Scholarship",badge:"UK"},{id:"chevening",label:"Chevening Scholarship",badge:"UK"},{id:"acs",label:"ASEAN Scholarships",badge:"Singapore"},{id:"nas",label:"NUS/NTU Merit Scholarship",badge:"Singapore"},{id:"australia-awards",label:"Australia Awards",badge:"Australia"},{id:"endeavour",label:"Endeavour Leadership",badge:"Australia"}];
  const toggleS=(id)=>{const n=scholarships.includes(id)?scholarships.filter(s=>s!==id):[...scholarships,id];setScholarships(n);emit({scholarships_interest:n});};
  const emit=(patch)=>onChange?.({budget_currency:currency,total_budget:totalBudget,annual_tuition_budget:annualTuition,annual_living_budget:livingCost,funding_source:fundingSource,scholarships_interest:scholarships,...patch});
  const totalLKR=toLKR(parseFloat(totalBudget)||0,currency);
  return(
    <div>
      {totalBudget&&<div className="fin-summary-bar"><div><div className="fin-sum-label">Total Budget</div><div className={`fin-sum-val${totalLKR<5000000?" warn":""}`}>{formatLKR(totalLKR)}</div></div>{totalLKR<5000000&&<Alert type="warn">Budget may be low for overseas study.</Alert>}</div>}
      <div className="fin-panel">
        <div className="slabel" style={{marginTop:0}}>Budget</div>
        <div className="fgrid">
          <div className="field"><label className="flabel">Total Budget <span className="req">*</span></label><div className="currency-select"><select value={currency} onChange={e=>{setCurrency(e.target.value);emit({budget_currency:e.target.value});}}>{["LKR","USD","EUR","GBP","AUD","SGD"].map(c=><option key={c}>{c}</option>)}</select><input type="number" value={totalBudget} onChange={e=>{setTotalBudget(e.target.value);emit({total_budget:e.target.value});}} placeholder="Amount" /></div></div>
          <div className="field"><label className="flabel">Funding Source</label><select value={fundingSource} onChange={e=>{setFundingSource(e.target.value);emit({funding_source:e.target.value});}}><option value="">— Select —</option>{["Personal / Family Savings","Bank Loan","Sponsor / Employer","Scholarship (Full)","Scholarship (Partial) + Self","Government Grant"].map(o=><option key={o}>{o}</option>)}</select></div>
          <div className="field"><label className="flabel">Annual Tuition Budget</label><input type="number" value={annualTuition} onChange={e=>{setAnnualTuition(e.target.value);emit({annual_tuition_budget:e.target.value});}} placeholder={`Per year (${currency})`} /></div>
          <div className="field"><label className="flabel">Annual Living Budget</label><input type="number" value={livingCost} onChange={e=>{setLivingCost(e.target.value);emit({annual_living_budget:e.target.value});}} placeholder={`Per year (${currency})`} /></div>
        </div>
      </div>
      <div className="fin-panel">
        <div className="slabel" style={{marginTop:0}}>Scholarships of Interest</div>
        {SCHOLARSHIPS.map(s=>(<div key={s.id} className="scholarship-row" onClick={()=>toggleS(s.id)}><input type="checkbox" checked={scholarships.includes(s.id)} onChange={()=>{}} /><span className="scholarship-label">{s.label}</span><span className="scholarship-badge">{s.badge}</span></div>))}
      </div>
    </div>
  );
}

// ── PROFILE STEP ─────────────────────────────────────────────────────────────
function ProfileStep({data,onNext}){
  const [activeTab,setActiveTab]=useState("personal");
  const [form,setForm]=useState({full_name:"",email:"",phone:"",dob:"",gender:"",nic:"",address:"",district:"",country:"",program_interest:"",study_level:"",current_qualification:"",stream:"",...data});
  const [financial,setFinancial]=useState(data.financial||{});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const SL_DISTRICTS=["Colombo","Gampaha","Kalutara","Kandy","Matale","Nuwara Eliya","Galle","Matara","Hambantota","Jaffna","Kilinochchi","Mannar","Mullaitivu","Vavuniya","Puttalam","Kurunegala","Anuradhapura","Polonnaruwa","Badulla","Monaragala","Ratnapura","Kegalle","Trincomalee","Batticaloa","Ampara"];
  const submit=()=>{
    if(!form.full_name||!form.email||!form.phone||!form.country||!form.current_qualification){setErr("Complete all required fields (*) in the Personal tab.");return;}
    if(!financial.total_budget){setErr("Enter your total budget in the Financial tab.");return;}
    setErr("");onNext({...form,financial});
  };
  return(
    <div className="fade-up">
      <div className="panel">
        <div className="panel-title">Your Profile</div>
        <div className="panel-sub">Complete both sections for accurate recommendations</div>
        <div className="profile-tabs">
          <button className={`ptab${activeTab==="personal"?" active":""}`} onClick={()=>setActiveTab("personal")}> Personal</button>
          <button className={`ptab${activeTab==="financial"?" active":""}`} onClick={()=>setActiveTab("financial")}> Financial</button>
        </div>
        {activeTab==="personal"&&(
          <div>
            <div className="slabel" style={{marginTop:0}}>Personal Information</div>
            <div className="fgrid">
              <div className="field"><label className="flabel">Full Name <span className="req">*</span></label><input value={form.full_name} onChange={e=>f("full_name",e.target.value)} placeholder="e.g. Kasun Sampath Perera" /></div>
              <div className="field"><label className="flabel">Email <span className="req">*</span></label><input value={form.email} onChange={e=>f("email",e.target.value)} /></div>
              <div className="field"><label className="flabel">Phone <span className="req">*</span></label><input value={form.phone} onChange={e=>f("phone",e.target.value)} placeholder="+94 77 000 0000" /></div>
              <div className="field"><label className="flabel">Date of Birth</label><input type="date" value={form.dob} onChange={e=>f("dob",e.target.value)} /></div>
              <div className="field"><label className="flabel">Gender</label><select value={form.gender} onChange={e=>f("gender",e.target.value)}><option value="">— Select —</option>{["Male","Female","Prefer not to say"].map(g=><option key={g}>{g}</option>)}</select></div>
              <div className="field"><label className="flabel">NIC / Passport No.</label><input value={form.nic} onChange={e=>f("nic",e.target.value)} /></div>
              <div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Home Address</label><input value={form.address} onChange={e=>f("address",e.target.value)} placeholder="No., Street, City" /></div>
              <div className="field"><label className="flabel">District</label><select value={form.district} onChange={e=>f("district",e.target.value)}><option value="">— Select —</option>{SL_DISTRICTS.map(d=><option key={d}>{d}</option>)}</select></div>
              <div className="field"><label className="flabel">Study Destination <span className="req">*</span></label><select value={form.country} onChange={e=>f("country",e.target.value)}><option value="">— Select —</option>{["UK","Singapore","Australia"].map(c=><option key={c}>{c}</option>)}</select></div>
            </div>
            <div className="slabel">Education</div>
            <div className="fgrid">
              <div className="field"><label className="flabel">Program Interest</label><select value={form.program_interest} onChange={e=>f("program_interest",e.target.value)}><option value="">— Select —</option>{["Engineering","Computer Science","Business","Medicine","Science","Arts","Law","Other"].map(p=><option key={p}>{p}</option>)}</select></div>
              <div className="field"><label className="flabel">Study Level</label><select value={form.study_level} onChange={e=>f("study_level",e.target.value)}><option value="">— Select —</option>{["Bachelor's Degree","Master's Degree","PhD"].map(l=><option key={l}>{l}</option>)}</select></div>
              <div className="field"><label className="flabel">Highest Qualification <span className="req">*</span></label><select value={form.current_qualification} onChange={e=>{f("current_qualification",e.target.value);f("stream","");}}><option value="">— Select —</option>{["GCE A/L","Diploma","Bachelor's Degree","Master's Degree"].map(q=><option key={q}>{q}</option>)}</select></div>
              {form.current_qualification==="GCE A/L"&&(<div className="field"><label className="flabel">A/L Stream</label><select value={form.stream} onChange={e=>f("stream",e.target.value)}><option value="">— Select —</option>{["Physical Science","Bio Science","Commerce","Arts","Technology"].map(s=><option key={s}>{s}</option>)}</select></div>)}
            </div>
          </div>
        )}
        {activeTab==="financial"&&<FinancialSection value={financial} onChange={setFinancial} />}
        {err&&<Alert type="error">{err}</Alert>}
      </div>
      <div className="btn-row btn-row-end">
        <button className="btn btn-primary" onClick={submit}>Next: Documents →</button>
      </div>
    </div>
  );
}

// ── PIPELINE DISPLAY ────────────────────────────────────────────────────────
const AI_PIPELINE=[
  {id:"read",     label:"File decode & validation"                },
  {id:"preproc",  label:"OpenCV preprocessing (CLAHE · deskew)"   },
  {id:"ocr",      label:"OCR text extraction (Tesseract · EasyOCR fallback)"},
  {id:"classify", label:"TF-IDF + Naive Bayes document classifier"},
  {id:"extract",  label:"Regex field extraction (9 doc types)"},
  {id:"verify",   label:"ML confidence scoring"},
];

function PipelineView({stages,stageStatus}){
  return <div className="pipeline">{stages.map((s,i)=>{const st=stageStatus[i]||"wait";const cls=st==="running"?"ps-active":st==="done"?"ps-done":st==="error"?"":"";const statusCls=st==="running"?"ps-run":st==="done"?"ps-ok":st==="error"?"ps-err":"ps-wait";const label=st==="running"?"Processing…":st==="done"?"Done":st==="error"?"Error":"Waiting";return <div className={`pipe-step ${cls}`} key={s.id}><span className="pipe-icon">{s.icon}</span><span className="pipe-label">{s.label}</span><span className={`pipe-status ${statusCls}`}>{st==="running"&&<span className="spin" style={{marginRight:".3rem"}}/>}{label}</span></div>;})}</div>;
}

// ── MANUAL FORMS (unchanged from original) ──────────────────────────────────
const STREAM_SUBJECTS={"Physical Science":["Combined Maths","Physics","Chemistry"],"Bio Science":["Biology","Chemistry","Physics"],"Commerce":["Business Studies","Accounting","Economics"],"Arts":["Economics","History","Geography"],"Technology":["Engineering Technology","Science for Technology","ICT"]};

// CHANGE v6: ALevelManualForm — english_proficiency / EnglishSection REMOVED.
//            User must upload IELTS/TOEFL/PTE as a separate document type.
function ALevelManualForm({stream,onSubmit,onBack}){
  const subjects=STREAM_SUBJECTS[stream]||["Subject 1","Subject 2","Subject 3"];
  const [grades,setGrades]=useState({});
  const [indexNo,setIndexNo]=useState("");
  const [zScore,setZScore]=useState("");
  const [err,setErr]=useState("");
  const submit=()=>{
    if(subjects.some(s=>!grades[s])){setErr("Select a grade for every subject.");return;}
    setErr("");
    // ✕ english_proficiency NOT included — must be uploaded as a separate document
    onSubmit({document_type:"A-Level Results",subjects:grades,index_number:indexNo,z_score:zScore||"N/A"});
  };
  return(
    <div>
      <Alert type="info">English proficiency (IELTS/TOEFL/PTE) is a <strong>separate document</strong>. Select the relevant tab to upload it after submitting your A/L results.</Alert>
      <div className="slabel" style={{marginTop:"1rem"}}>Subject Grades</div>
      {subjects.map(sub=>(<div key={sub} style={{marginBottom:"1.1rem"}}><div className="flabel" style={{marginBottom:".4rem"}}>{sub}</div><div className="grade-row">{["A","B","C","S","F"].map(g=>(<button key={g} className={`gpill${grades[sub]===g?` g${g}`:""}`} onClick={()=>setGrades(p=>({...p,[sub]:g}))}>{g}</button>))}</div></div>))}
      <div className="fgrid" style={{marginTop:"1rem"}}>
        <div className="field"><label className="flabel">Index Number</label><input value={indexNo} onChange={e=>setIndexNo(e.target.value)} /></div>
        <div className="field"><label className="flabel">Z-Score (optional)</label><input value={zScore} onChange={e=>setZScore(e.target.value)} /></div>
      </div>
      {err&&<Alert type="error">{err}</Alert>}
      <div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div>
    </div>
  );
}

function DegreeManualForm({docType,data={},onSubmit,onBack}){
  const isMasters=docType==="Master's Degree";
  const [form,setForm]=useState({university_name:"",degree_program:"",graduation_year:"",gpa_system:"",gpa_value:"",degree_class:"",thesis_title:"",...data});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));const years=Array.from({length:25},(_,i)=>2024-i);
  const submit=()=>{if(!form.university_name||!form.degree_program||!form.graduation_year||!form.gpa_system||!form.gpa_value){setErr("Fill all required fields.");return;}onSubmit({document_type:docType,...form,graduation_year:+form.graduation_year,gpa_normalized:normalizeGpa(form.gpa_value,form.gpa_system)});};
  return(<div><div className="fgrid"><div className="field"><label className="flabel">University <span className="req">*</span></label><input value={form.university_name} onChange={e=>f("university_name",e.target.value)} /></div><div className="field"><label className="flabel">Degree Program <span className="req">*</span></label><input value={form.degree_program} onChange={e=>f("degree_program",e.target.value)} /></div>{isMasters&&<div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Thesis Title</label><input value={form.thesis_title} onChange={e=>f("thesis_title",e.target.value)} /></div>}<div className="field"><label className="flabel">Graduation Year <span className="req">*</span></label><select value={form.graduation_year} onChange={e=>f("graduation_year",e.target.value)}><option value="">— Select —</option>{years.map(y=><option key={y}>{y}</option>)}</select></div><div className="field"><label className="flabel">Grading System <span className="req">*</span></label><select value={form.gpa_system} onChange={e=>{f("gpa_system",e.target.value);f("gpa_value","");}}><option value="">— Select —</option>{["GPA (4.0 scale)","GPA (5.0 scale)","UK Class","Percentage"].map(s=><option key={s}>{s}</option>)}</select></div>{form.gpa_system==="GPA (4.0 scale)"&&<div className="field"><label className="flabel">GPA <span className="req">*</span></label><input type="number" min={0} max={4} step={0.01} value={form.gpa_value} onChange={e=>f("gpa_value",e.target.value)} placeholder="0.00–4.00" /></div>}{form.gpa_system==="GPA (5.0 scale)"&&<div className="field"><label className="flabel">GPA <span className="req">*</span></label><input type="number" min={0} max={5} step={0.01} value={form.gpa_value} onChange={e=>f("gpa_value",e.target.value)} placeholder="0.00–5.00" /></div>}{form.gpa_system==="UK Class"&&<div className="field"><label className="flabel">Classification <span className="req">*</span></label><select value={form.gpa_value} onChange={e=>f("gpa_value",e.target.value)}><option value="">— Select —</option>{["First Class","Upper Second (2:1)","Lower Second (2:2)","Third Class"].map(c=><option key={c}>{c}</option>)}</select></div>}{form.gpa_system==="Percentage"&&<div className="field"><label className="flabel">Percentage <span className="req">*</span></label><input type="number" min={0} max={100} value={form.gpa_value} onChange={e=>f("gpa_value",e.target.value)} /></div>}</div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function DiplomaManualForm({onSubmit,onBack}){
  const [form,setForm]=useState({student_name:"",institution:"",program:"",grade:"",completion_year:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const years=Array.from({length:20},(_,i)=>2024-i);
  const submit=()=>{if(!form.student_name||!form.institution||!form.program){setErr("Fill name, institution and program.");return;}setErr("");onSubmit({document_type:"Diploma",...form});};
  return(<div><div className="fgrid"><div className="field"><label className="flabel">Student Name <span className="req">*</span></label><input value={form.student_name} onChange={e=>f("student_name",e.target.value)} /></div><div className="field"><label className="flabel">Institution <span className="req">*</span></label><input value={form.institution} onChange={e=>f("institution",e.target.value)} placeholder="e.g. SLIIT, NIBM, NSBM" /></div><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Program / Diploma Name <span className="req">*</span></label><input value={form.program} onChange={e=>f("program",e.target.value)} placeholder="e.g. Diploma in IT" /></div><div className="field"><label className="flabel">Grade</label><select value={form.grade} onChange={e=>f("grade",e.target.value)}><option value="">— Select —</option>{["Distinction","Merit","Pass"].map(g=><option key={g}>{g}</option>)}</select></div><div className="field"><label className="flabel">Completion Year</label><select value={form.completion_year} onChange={e=>f("completion_year",e.target.value)}><option value="">— Select —</option>{years.map(y=><option key={y}>{y}</option>)}</select></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function IELTSManualForm({onSubmit,onBack}){
  const BANDS=new Set([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9]);
  const [form,setForm]=useState({candidate_name:"",overall:"",listening:"",reading:"",writing:"",speaking:"",test_date:"",trf_number:"",test_centre:"",nationality:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const validBand=(v)=>BANDS.has(parseFloat(v));
  const submit=()=>{if(!form.candidate_name||!form.overall){setErr("Name and overall band are required.");return;}if(!validBand(form.overall)){setErr("Overall band must be 0–9 in 0.5 steps (e.g. 6.5).");return;}setErr("");onSubmit({document_type:"IELTS Certificate",...form});};
  const secs=["overall","listening","reading","writing","speaking"];
  return(<div><div className="fgrid"><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Candidate Name <span className="req">*</span></label><input value={form.candidate_name} onChange={e=>f("candidate_name",e.target.value)} /></div></div><div className="slabel" style={{marginTop:"1rem"}}>Band Scores (0–9, half-band steps)</div><div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:".5rem",marginBottom:"1rem"}}>{secs.map(k=>{const v=parseFloat(form[k])||0;const ok=form[k]===""||validBand(form[k]);return(<div key={k} className="field"><label className="flabel">{k.charAt(0).toUpperCase()+k.slice(1)}{k==="overall"&&<span className="req"> *</span>}</label><input className="band-input" type="number" min={0} max={9} step={0.5} value={form[k]} placeholder="0.0" onChange={e=>f(k,e.target.value)} style={{borderColor:form[k]&&!ok?"var(--red)":undefined}} />{form[k]!=""&&<div style={{fontFamily:"var(--mono)",fontSize:".6rem",color:ok?"var(--green)":"var(--red)",textAlign:"center",marginTop:".2rem"}}>{ok?bandStatus(v):"Invalid"}</div>}</div>);})}</div><div className="fgrid"><div className="field"><label className="flabel">Test Date</label><input value={form.test_date} onChange={e=>f("test_date",e.target.value)} placeholder="DD/MM/YYYY" /></div><div className="field"><label className="flabel">TRF Number</label><input value={form.trf_number} onChange={e=>f("trf_number",e.target.value)} /></div><div className="field"><label className="flabel">Test Centre</label><input value={form.test_centre} onChange={e=>f("test_centre",e.target.value)} /></div><div className="field"><label className="flabel">Nationality</label><input value={form.nationality} onChange={e=>f("nationality",e.target.value)} placeholder="Sri Lankan" /></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function TOEFLManualForm({onSubmit,onBack}){
  const [form,setForm]=useState({candidate_name:"",total:"",reading:"",listening:"",speaking:"",writing:"",test_date:"",registration_number:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const submit=()=>{if(!form.candidate_name||!form.total){setErr("Name and total score are required.");return;}const t=parseInt(form.total);if(isNaN(t)||t<0||t>120){setErr("Total score must be 0–120.");return;}setErr("");onSubmit({document_type:"TOEFL Certificate",...form});};
  const secs=[["Reading","reading",30],["Listening","listening",30],["Speaking","speaking",30],["Writing","writing",30]];
  return(<div><div className="fgrid"><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Candidate Name <span className="req">*</span></label><input value={form.candidate_name} onChange={e=>f("candidate_name",e.target.value)} /></div><div className="field"><label className="flabel">Total Score (0–120) <span className="req">*</span></label><input type="number" min={0} max={120} value={form.total} onChange={e=>f("total",e.target.value)} /></div>{secs.map(([label,key,max])=>(<div key={key} className="field"><label className="flabel">{label} (0–{max})</label><input type="number" min={0} max={max} value={form[key]} onChange={e=>f(key,e.target.value)} /></div>))}<div className="field"><label className="flabel">Test Date</label><input value={form.test_date} onChange={e=>f("test_date",e.target.value)} placeholder="DD/MM/YYYY" /></div><div className="field"><label className="flabel">Registration Number</label><input value={form.registration_number} onChange={e=>f("registration_number",e.target.value)} /></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function PTEManualForm({onSubmit,onBack}){
  const [form,setForm]=useState({candidate_name:"",overall:"",listening:"",reading:"",writing:"",speaking:"",test_date:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const submit=()=>{if(!form.candidate_name||!form.overall){setErr("Name and overall score are required.");return;}const o=parseInt(form.overall);if(isNaN(o)||o<10||o>90){setErr("Overall score must be 10–90.");return;}setErr("");onSubmit({document_type:"PTE Certificate",...form});};
  const secs=[["Listening","listening"],["Reading","reading"],["Writing","writing"],["Speaking","speaking"]];
  return(<div><div className="fgrid"><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Candidate Name <span className="req">*</span></label><input value={form.candidate_name} onChange={e=>f("candidate_name",e.target.value)} /></div><div className="field"><label className="flabel">Overall Score (10–90) <span className="req">*</span></label><input type="number" min={10} max={90} value={form.overall} onChange={e=>f("overall",e.target.value)} /></div>{secs.map(([label,key])=>(<div key={key} className="field"><label className="flabel">{label} (10–90)</label><input type="number" min={10} max={90} value={form[key]} onChange={e=>f(key,e.target.value)} /></div>))}<div className="field"><label className="flabel">Test Date</label><input value={form.test_date} onChange={e=>f("test_date",e.target.value)} placeholder="DD/MM/YYYY" /></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function PassportManualForm({onSubmit,onBack}){
  const [form,setForm]=useState({surname:"",given_names:"",passport_number:"",nationality:"Sri Lankan",date_of_birth:"",sex:"",place_of_birth:"",issue_date:"",expiry_date:"",issuing_authority:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const submit=()=>{if(!form.surname||!form.given_names||!form.expiry_date){setErr("Surname, given names and expiry date are required.");return;}setErr("");const masked=form.passport_number.length>4?"*".repeat(form.passport_number.length-4)+form.passport_number.slice(-4):form.passport_number;onSubmit({document_type:"Passport",...form,passport_number:masked});};
  return(<div><Alert type="warn">Passport number will be masked for privacy — only last 4 digits stored.</Alert><div className="fgrid" style={{marginTop:"1rem"}}><div className="field"><label className="flabel">Surname <span className="req">*</span></label><input value={form.surname} onChange={e=>f("surname",e.target.value.toUpperCase())} placeholder="PERERA" /></div><div className="field"><label className="flabel">Given Names <span className="req">*</span></label><input value={form.given_names} onChange={e=>f("given_names",e.target.value)} placeholder="Kamal Suresh" /></div><div className="field"><label className="flabel">Passport Number</label><input value={form.passport_number} onChange={e=>f("passport_number",e.target.value.toUpperCase())} placeholder="N1234567" /></div><div className="field"><label className="flabel">Nationality</label><input value={form.nationality} onChange={e=>f("nationality",e.target.value)} /></div><div className="field"><label className="flabel">Date of Birth</label><input value={form.date_of_birth} onChange={e=>f("date_of_birth",e.target.value)} placeholder="DD/MM/YYYY" /></div><div className="field"><label className="flabel">Sex</label><select value={form.sex} onChange={e=>f("sex",e.target.value)}><option value="">— Select —</option><option value="M">Male</option><option value="F">Female</option></select></div><div className="field"><label className="flabel">Place of Birth</label><input value={form.place_of_birth} onChange={e=>f("place_of_birth",e.target.value)} placeholder="Colombo" /></div><div className="field"><label className="flabel">Issuing Authority</label><input value={form.issuing_authority} onChange={e=>f("issuing_authority",e.target.value)} placeholder="Dept. of Immigration" /></div><div className="field"><label className="flabel">Issue Date</label><input value={form.issue_date} onChange={e=>f("issue_date",e.target.value)} placeholder="DD/MM/YYYY" /></div><div className="field"><label className="flabel">Expiry Date <span className="req">*</span></label><input value={form.expiry_date} onChange={e=>f("expiry_date",e.target.value)} placeholder="DD/MM/YYYY" /></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

function BankManualForm({onSubmit,onBack}){
  const [form,setForm]=useState({account_holder:"",bank_name:"",account_number:"",currency:"LKR",opening_balance:"",closing_balance:"",statement_period:""});
  const [err,setErr]=useState("");
  const f=(k,v)=>setForm(p=>({...p,[k]:v}));
  const SL_BANKS=["Bank of Ceylon","People's Bank","Commercial Bank","Hatton National Bank","Sampath Bank","Seylan Bank","Nations Trust Bank","DFCC Bank","Pan Asia Bank","NDB Bank","NSB","Citi Bank","HSBC","Standard Chartered"];
  const submit=()=>{if(!form.account_holder||!form.bank_name||!form.closing_balance){setErr("Account holder, bank name and closing balance are required.");return;}setErr("");const masked=form.account_number.length>4?"****"+form.account_number.slice(-4):form.account_number;onSubmit({document_type:"Financial Statement",...form,account_number:masked});};
  return(<div><div className="fgrid"><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Account Holder Name <span className="req">*</span></label><input value={form.account_holder} onChange={e=>f("account_holder",e.target.value)} /></div><div className="field"><label className="flabel">Bank Name <span className="req">*</span></label><select value={form.bank_name} onChange={e=>f("bank_name",e.target.value)}><option value="">— Select Bank —</option>{SL_BANKS.map(b=><option key={b}>{b}</option>)}<option value="Other">Other</option></select></div>{form.bank_name==="Other"&&<div className="field"><label className="flabel">Bank Name (Other)</label><input onChange={e=>f("bank_name",e.target.value)} placeholder="Enter bank name" /></div>}<div className="field"><label className="flabel">Account Number (last 4 digits shown)</label><input value={form.account_number} onChange={e=>f("account_number",e.target.value)} placeholder="Full account number" /></div><div className="field"><label className="flabel">Currency</label><select value={form.currency} onChange={e=>f("currency",e.target.value)}>{["LKR","USD","EUR","GBP","AUD","SGD"].map(c=><option key={c}>{c}</option>)}</select></div><div className="field"><label className="flabel">Opening Balance</label><input value={form.opening_balance} onChange={e=>f("opening_balance",e.target.value)} placeholder="e.g. 500,000.00" /></div><div className="field"><label className="flabel">Closing Balance <span className="req">*</span></label><input value={form.closing_balance} onChange={e=>f("closing_balance",e.target.value)} placeholder="e.g. 1,250,000.00" /></div><div className="field" style={{gridColumn:"1/-1"}}><label className="flabel">Statement Period</label><input value={form.statement_period} onChange={e=>f("statement_period",e.target.value)} placeholder="e.g. January 2024 – June 2024" /></div></div>{err&&<Alert type="error">{err}</Alert>}<div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Back</button><button className="btn btn-primary" onClick={submit}>Check Eligibility →</button></div></div>);
}

// ── DOC STEP — v6: all 9 tabs, checklist panel, eng-next banner ─────────────

// All 9 document type definitions with categories for the checklist
const DOC_TYPE_DEFS = {
  "A-Level Results":     {cat:"academic"},
  "Bachelor's Degree":   { cat:"academic"},
  "Master's Degree":     { cat:"academic"},
  "Diploma":             {cat:"academic"},
  "IELTS Certificate":   {cat:"english"},
  "TOEFL Certificate":   {cat:"english"},
  "PTE Certificate":     {cat:"english"},
  "Passport":            {cat:"identity"},
  "Financial Statement": {cat:"financial"},
};

const DOC_CATS = {
  academic:  {label:"Academic",            types:["A-Level Results","Bachelor's Degree","Master's Degree","Diploma","IELTS Certificate","TOEFL Certificate","PTE Certificate"]},
  identity:  {label:"Identity",            types:["Passport"]},
  financial: {label:"Financial",           types:["Financial Statement"]},
};

const DOC_TYPE_LIST = [
  {id:"A-Level Results",     label:"A/L Results"},
  {id:"Bachelor's Degree",   label:"Bachelor's"},
  {id:"Master's Degree",     label:"Master's"},
  {id:"Diploma",             label:"Diploma"},
  {id:"IELTS Certificate",   label:"IELTS"},
  {id:"TOEFL Certificate",   label:"TOEFL"},
  {id:"PTE Certificate",     label:"PTE"},
  {id:"Passport",            label:"Passport"},
  {id:"Financial Statement", label:"Bank Stmt"},
];

// NEW v6: Checklist panel shown above doc grid
function DocChecklist({uploadedDocs, onSelectType}) {
  const allTypes = Object.keys(DOC_TYPE_DEFS);
  const doneCount = Object.values(uploadedDocs).filter(Boolean).length;

  return (
    <div className="doc-checklist">
      <div className="doc-checklist-title">
         Document Checklist
        <span style={{marginLeft:"auto",fontFamily:"var(--mono)",fontSize:".65rem",
          color:"var(--text3)",fontWeight:400}}>
          {doneCount} / {allTypes.length} uploaded
        </span>
      </div>
      <div className="checklist-progress">
        <div className="checklist-progress-fill" style={{width:`${(doneCount/allTypes.length)*100}%`}} />
      </div>
      <div className="checklist-cats">
        {Object.entries(DOC_CATS).map(([cat, {label, types}]) => (
          <div key={cat}>
            <div className="checklist-cat-label">{label}</div>
            <div className="checklist-chips">
              {types.map(t => {
                const done = !!uploadedDocs[t];
                const def  = DOC_TYPE_DEFS[t];
                return (
                  <span
                    key={t}
                    className={`checklist-chip${done?" done":""}`}
                    style={{cursor:"pointer"}}
                    onClick={() => onSelectType(t)}
                  >
                    {done ? "✓" : "○"} {def.icon} {t}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {doneCount < allTypes.length && (
        <div className="checklist-remaining">
          Still needed: {allTypes.filter(t=>!uploadedDocs[t]).join(", ")}
        </div>
      )}
      {doneCount === allTypes.length && (
        <div className="checklist-complete"> All 9 documents uploaded!</div>
      )}
    </div>
  );
}



function DocumentStep({profile,docData,onNext,onBack,user}){
  const qual=profile.current_qualification;
  const defaultType=qual==="GCE A/L"?"A-Level Results":qual==="Bachelor's Degree"?"Bachelor's Degree":qual==="Master's Degree"?"Master's Degree":qual==="Diploma"?"Diploma":"A-Level Results";

  const [tab,setTab]=useState("upload");
  const [selectedType,setSelectedType]=useState(defaultType);

  // v6: track all uploaded docs so checklist + banner work
  const [uploadedDocs,setUploadedDocs]=useState(docData?.documents&&typeof docData.documents==="object"?docData.documents:{});
  const [pendingUploads,setPendingUploads]=useState([]);

  const fileRef=useRef();
  const [drag,setDrag]=useState(false);
  const [pipeStatus,setPipeStatus]=useState({});
  const [running,setRunning]=useState(false);
  const [aiResult,setAiResult]=useState(null);
  const [aiError,setAiError]=useState("");
  const [progress,setProgress]=useState("");
  const [eng,setEng]=useState(docData?.english_proficiency||{});
  const [serverDocs,setServerDocs]=useState([]);
  const [docsLoading,setDocsLoading]=useState(false);
  const [docsError,setDocsError]=useState("");
  const [docsBusyId,setDocsBusyId]=useState("");

  // A/L never needs inline english section (it's a separate doc type)
  const needsEng = ["Bachelor's Degree","Master's Degree","Diploma"].includes(selectedType);

  // Show eng-next banner when A/L is done but no English cert uploaded yet
  const alDone    = !!uploadedDocs["A-Level Results"];
  const engDone   = ["IELTS Certificate","TOEFL Certificate","PTE Certificate"].some(t=>uploadedDocs[t]);
  const showEngBanner = alDone && !engDone;

  const revokePreview = (item) => {
    if(item?.preview) URL.revokeObjectURL(item.preview);
  };

  const refreshServerDocs = useCallback(async()=>{
    if(!user?.token) return;
    setDocsLoading(true);
    setDocsError("");
    try{
      const docs = await fetchUserDocuments(user.token);
      setServerDocs(docs);
    }catch(e){
      setDocsError(e?.message || "Unable to load documents");
    }finally{
      setDocsLoading(false);
    }
  },[user?.token]);

  useEffect(()=>{refreshServerDocs();},[refreshServerDocs]);

  const switchType = (t) => {
    setSelectedType(t);
    setAiResult(null);setAiError("");setPipeStatus({});
    setProgress("");
    // restore prior result for this type if already uploaded
    if(uploadedDocs[t]) setAiResult({data:uploadedDocs[t],confidence:0.75,ocrEngine:"stored"});
  };

  const handleFiles=(files)=>{
    const nextFiles=Array.from(files||[]).filter(Boolean);
    if(!nextFiles.length)return;
    setAiResult(null);
    setAiError("");
    setPipeStatus({});
    setProgress("");
    setPendingUploads((current)=>[
      ...current,
      ...nextFiles.map((uploadFile,index)=>(
        {
          id:`${Date.now()}-${index}-${Math.random().toString(36).slice(2,8)}`,
          file:uploadFile,
          assignedType:API_DOC_TYPE_MAP[selectedType]||"auto",
          preview:uploadFile.type.startsWith("image")?URL.createObjectURL(uploadFile):null,
        }
      )),
    ]);
  };

  const updatePendingType=(id,assignedType)=>{
    setPendingUploads((current)=>current.map((item)=>item.id===id?{...item,assignedType}:item));
  };

  const removePendingUpload=(id)=>{
    setPendingUploads((current)=>{
      const item=current.find((entry)=>entry.id===id);
      revokePreview(item);
      return current.filter((entry)=>entry.id!==id);
    });
  };

  const runAIExtraction=async()=>{
    if(!pendingUploads.length)return;
    setRunning(true);setAiError("");setPipeStatus({});setAiResult(null);
    const setStage=(i,st)=>setPipeStatus(p=>({...p,[i]:st}));
    setStage(0,"running");await new Promise(r=>setTimeout(r,180));setStage(0,"done");
    setStage(1,"running");await new Promise(r=>setTimeout(r,250));setStage(1,"done");
    setStage(2,"running");setStage(3,"running");
    try{
      let lastResult=null;
      const uploadedResults={};
      for(let index=0;index<pendingUploads.length;index+=1){
        const pending=pendingUploads[index];
        const typeLabel=UPLOAD_DOC_TYPE_OPTIONS.find((opt)=>opt.value===pending.assignedType)?.label||"Auto identify";
        const result=await extractDocumentWithAI(pending.file,pending.assignedType,(msg,pct)=>{
          setProgress(`File ${index+1}/${pendingUploads.length} · ${pending.file.name} · ${typeLabel} · ${msg}`);
          if(pct>=45){setStage(2,"done");setStage(3,"running");}
          if(pct>=60){setStage(3,"done");setStage(4,"running");}
          if(pct>=75){setStage(4,"done");setStage(5,"running");}
        },user?.token);
        lastResult=result;
        uploadedResults[result.data.document_type]=result.data;
      }
      setStage(4,"done");setStage(5,"done");
      setUploadedDocs(p=>({...p,...uploadedResults}));
      if(uploadedResults[selectedType]){
        setAiResult({data:uploadedResults[selectedType],confidence:lastResult?.confidence??0.75,ocrEngine:lastResult?.ocrEngine||"ocr"});
      }else if(lastResult){
        setAiResult(lastResult);
        if(lastResult?.data?.document_type) setSelectedType(lastResult.data.document_type);
      }
      setPendingUploads((current)=>{
        current.forEach(revokePreview);
        return [];
      });
      refreshServerDocs();
    }catch(e){
      setStage(2,"error");setStage(3,"error");setStage(4,"error");
      setAiError(e.message);
    }
    setRunning(false);setProgress("");
  };

  const buildEligibilityPayload = (baseDocs)=>{
    const docs={...baseDocs};
    if(aiResult?.data?.document_type){
      const currentDoc={...aiResult.data};
      if(currentDoc.gpa_value&&!currentDoc.gpa_normalized){
        currentDoc.gpa_normalized=parseFloat(currentDoc.gpa_value)||0;
      }
      docs[currentDoc.document_type]=currentDoc;
    }
    return {
      documents:docs,
      primary_document_type:selectedType,
      english_proficiency:eng,
    };
  };

  const proceed=()=>{
    const finalPayload=buildEligibilityPayload(uploadedDocs);
    if(!Object.keys(finalPayload.documents||{}).length){
      setAiError("Add at least one extracted or manually entered document before eligibility check.");
      return;
    }
    onNext(finalPayload);
  };

  // Manual submit also marks type as uploaded
  const handleManualSubmit = (doc) => {
    const manualDoc={...doc};
    if(manualDoc.gpa_value&&!manualDoc.gpa_normalized){
      manualDoc.gpa_normalized=parseFloat(manualDoc.gpa_value)||0;
    }
    const nextDocs={...uploadedDocs,[selectedType]:manualDoc};
    setUploadedDocs(nextDocs);
    onNext(buildEligibilityPayload(nextDocs));
  };

  return(
    <div className="fade-up">

      {/* v6: Checklist panel — always visible */}
      <DocChecklist uploadedDocs={uploadedDocs} onSelectType={switchType} />

      {/* v6: Eng-next banner — shown after A/L is done, before any English cert */}


      <div className="panel">
        <div className="panel-title">Academic & Supporting Documents</div>
        <div className="panel-sub">Upload for OCR + ML extraction (all 9 document types) or enter manually</div>
        <div className="tabs">
          <button className={`tab${tab==="upload"?" active":""}`} onClick={()=>setTab("upload")}> OCR Upload & Extract</button>
          <button className={`tab${tab==="manual"?" active":""}`} onClick={()=>setTab("manual")}>Manual Entry</button>
        </div>

        <div className="slabel">My Stored Documents</div>
        {docsError&&<Alert type="warn">{docsError}</Alert>}
        <div style={{display:"grid",gap:".5rem",marginBottom:"1rem"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:".75rem",flexWrap:"wrap"}}>
            <span style={{fontFamily:"var(--mono)",fontSize:".7rem",color:"var(--text3)"}}>{docsLoading?"Loading...":`${serverDocs.length} file(s) saved for this user`}</span>
            <button className="btn btn-ghost btn-sm" onClick={refreshServerDocs} disabled={docsLoading}>↻ Refresh</button>
          </div>
          {!serverDocs.length && !docsLoading && (
            <div style={{padding:".75rem .9rem",border:"1px dashed var(--border2)",borderRadius:"var(--r)",fontFamily:"var(--mono)",fontSize:".68rem",color:"var(--text3)"}}>
              No stored documents yet. Upload files while logged in to save them.
            </div>
          )}
          {serverDocs.slice(0,8).map((doc)=>{
            const docId = doc.document_id || "";
            const busy = docsBusyId===docId;
            return (
              <div key={docId} style={{display:"grid",gridTemplateColumns:"1fr auto",gap:".6rem",alignItems:"center",padding:".6rem .75rem",background:"var(--bg3)",border:"1px solid var(--border)",borderRadius:"var(--r)"}}>
                <div style={{minWidth:0}}>
                  <div style={{fontSize:".84rem",fontWeight:700,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{doc.filename || "document"}</div>
                  <div style={{fontFamily:"var(--mono)",fontSize:".62rem",color:"var(--text3)",marginTop:".2rem"}}>{doc.content_type || "file"} · {Math.round((doc.file_size||0)/1024)} KB</div>
                </div>
                <div style={{display:"flex",gap:".45rem",flexWrap:"wrap",justifyContent:"flex-end"}}>
                  <button className="btn btn-ghost btn-sm" disabled={busy} onClick={async()=>{try{setDocsBusyId(docId);await openUserDocument(docId,user?.token);}catch(e){setDocsError(e?.message||"Open failed");}finally{setDocsBusyId("");}}}>Open</button>
                  <button className="btn btn-danger btn-sm" disabled={busy} onClick={async()=>{try{setDocsBusyId(docId);setDocsError("");await deleteUserDocument(docId,user?.token);setServerDocs((current)=>current.filter((d)=>d.document_id!==docId));}catch(e){setDocsError(e?.message||"Delete failed");}finally{setDocsBusyId("");}}}>Delete</button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Document type selector removed - using auto-identification instead */}
        {/* Stored documents and file upload sections only */}

        {tab==="upload"&&(
          <div>
            <Alert type="info">Students can upload multiple files here. Tag each file with a document type or leave it on <strong>Auto identify</strong> to use the backend classifier.</Alert>
            <input ref={fileRef} type="file" multiple accept=".png,.jpg,.jpeg,.pdf" style={{display:"none"}} onChange={e=>{handleFiles(e.target.files);e.target.value="";}} />
            {!pendingUploads.length?(
              <div className={`upload-zone${drag?" drag":""}`} onClick={()=>fileRef.current?.click()} onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);handleFiles(e.dataTransfer.files);}} style={{marginTop:"1rem"}}>
                <div className="upload-icon">{DOC_TYPE_LIST.find(d=>d.id===selectedType)?.icon||""}</div>
                <div className="upload-title">Drop one or more documents here</div>
                <div className="upload-sub">PNG · JPG · JPEG · PDF · Max 10 MB each</div>
              </div>
            ):(
              <div style={{marginTop:"1rem"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:".75rem",marginBottom:"1rem",flexWrap:"wrap"}}>
                  <span style={{fontFamily:"var(--mono)",fontSize:".75rem",color:"var(--text3)"}}>{pendingUploads.length} file(s) ready for extraction</span>
                  <div style={{display:"flex",gap:".5rem",flexWrap:"wrap"}}>
                    <button className="btn btn-ghost btn-sm" onClick={()=>fileRef.current?.click()}>+ Add More</button>
                    <button className="btn btn-ghost btn-sm" onClick={()=>{setPendingUploads((current)=>{current.forEach(revokePreview);return [];});setAiResult(null);setAiError("");setPipeStatus({});setProgress("");}}>Clear Queue</button>
                  </div>
                </div>
                <div style={{display:"grid",gap:".75rem"}}>
                  {pendingUploads.map((item)=>{
                    const isImage=!!item.preview;
                    return(
                      <div key={item.id} style={{display:"grid",gridTemplateColumns:isImage?"84px 1fr auto":"1fr auto",gap:".75rem",alignItems:"center",padding:".85rem",background:"var(--bg3)",border:"1px solid var(--border)",borderRadius:"var(--r)"}}>
                        {isImage&&<img src={item.preview} alt={item.file.name} className="upload-preview" style={{margin:0,maxHeight:"72px",width:"84px"}} />}
                        <div style={{minWidth:0}}>
                          <div style={{fontWeight:700,fontSize:".9rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{item.file.name}</div>
                          <div style={{fontFamily:"var(--mono)",fontSize:".68rem",color:"var(--text3)",margin:".2rem 0 .55rem"}}>{(item.file.size/1024).toFixed(0)} KB</div>
                          <select value={item.assignedType} onChange={e=>updatePendingType(item.id,e.target.value)}>
                            {UPLOAD_DOC_TYPE_OPTIONS.map((option)=><option key={option.value} value={option.value}>{option.label}</option>)}
                          </select>
                        </div>
                        <button className="btn btn-ghost btn-sm" onClick={()=>removePendingUpload(item.id)}>✕ Remove</button>
                      </div>
                    );
                  })}
                </div>
                {!running&&<button className="btn btn-primary btn-full" onClick={runAIExtraction} style={{marginTop:"1rem"}}>🔍 Extract {pendingUploads.length} Document{pendingUploads.length>1?"s":""}</button>}
              </div>
            )}
            {aiError&&<div style={{marginTop:"1rem"}}><Alert type="error">{aiError}</Alert></div>}
            {(running||Object.keys(pipeStatus).length>0)&&(
              <div style={{marginTop:"1rem"}}>
                <div className="slabel">OCR Extraction Pipeline {running&&<span style={{fontFamily:"var(--mono)",fontSize:".65rem",color:"var(--amber)"}}>— processing</span>}</div>
                {progress&&<div style={{fontFamily:"var(--mono)",fontSize:".72rem",color:"var(--accent2)",marginBottom:".5rem",padding:".5rem .75rem",background:"var(--accent-dim)",borderRadius:"var(--r)"}}>{progress}</div>}
                <PipelineView stages={AI_PIPELINE} stageStatus={pipeStatus} />
              </div>
            )}
            {aiResult&&(
              <div style={{marginTop:"1.25rem"}}>
                <div className="slabel">Extracted Data</div>
                {aiResult.warnings&&aiResult.warnings.length>0&&aiResult.warnings.map((w,i)=><Alert key={i} type="warn">{w}</Alert>)}
                <ExtractedDisplay data={aiResult.data} confidence={aiResult.confidence} ocrEngine={aiResult.ocrEngine} />
                {needsEng&&<div style={{marginTop:"1.25rem"}}><EnglishSection value={eng} onChange={setEng} /></div>}
                <div className="btn-row" style={{marginTop:"1.5rem"}}>
                  <button className="btn btn-ghost" onClick={onBack}>← Profile</button>
                  <button className="btn btn-primary" onClick={proceed}>Check Eligibility →</button>
                </div>
              </div>
            )}
          </div>
        )}

        {tab==="manual"&&(
          <div>
            {selectedType==="A-Level Results"&&<ALevelManualForm stream={profile.stream} onSubmit={handleManualSubmit} onBack={onBack} />}
            {(selectedType==="Bachelor's Degree"||selectedType==="Master's Degree")&&<DegreeManualForm docType={selectedType} data={uploadedDocs[selectedType]||{}} onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="Diploma"&&<DiplomaManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="IELTS Certificate"&&<IELTSManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="TOEFL Certificate"&&<TOEFLManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="PTE Certificate"&&<PTEManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="Passport"&&<PassportManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
            {selectedType==="Financial Statement"&&<BankManualForm onSubmit={handleManualSubmit} onBack={onBack} />}
          </div>
        )}
      </div>
    </div>
  );
}

// ── ELIGIBILITY STEP ────────────────────────────────────────────────────────
function EligibilityStep({elig,docData,profile,onNext,onBack}){
  const docs = docData?.documents||{};
  const ieltsDoc = docs["IELTS Certificate"];
  const toeflDoc = docs["TOEFL Certificate"];
  const pteDoc = docs["PTE Certificate"];
  const eng=docData.english_proficiency||{
    ielts: ieltsDoc?.overall ? {overall: ieltsDoc.overall} : null,
    toefl: toeflDoc?.total ? parseInt(toeflDoc.total,10) : null,
    pte: pteDoc?.overall ? parseInt(pteDoc.overall,10) : null,
  };
  const fin=profile.financial||{};
  const tierCls={top:"tier-top",good:"tier-good",average:"tier-average",foundation:"tier-foundation"}[elig.eligibility_tier]||"tier-average";
  const totalBudgetLKR=toLKR(parseFloat(fin.total_budget)||0,fin.budget_currency||"LKR");
  const engBadges=[];
  if(eng.ielts)engBadges.push(<span key="ielts" className="eng-badge eb-ielts">IELTS {eng.ielts.overall}</span>);
  if(eng.toefl)engBadges.push(<span key="toefl" className="eng-badge eb-toefl">TOEFL {eng.toefl}</span>);
  if(eng.pte)engBadges.push(<span key="pte" className="eng-badge eb-pte">PTE {eng.pte}</span>);
  return(
    <div className="fade-up">
      <div className={`elig-hero${elig.eligible?" pass":" fail"}`}>
        <div className="elig-score">{elig.grade_point.toFixed(2)}<span style={{fontSize:"1.1rem",fontWeight:400,opacity:.4}}> /4.0</span></div>
        <div style={{marginTop:".4rem",fontFamily:"var(--sans)",fontSize:"1.1rem",fontWeight:700,color:elig.eligible?"var(--accent2)":"var(--amber)"}}>{elig.eligible?"✓ Eligible for admission":"⚠ Alternative pathways available"}</div>
        {elig.eligibility_tier&&<div className={`elig-tier ${tierCls}`}>{elig.eligibility_tier.toUpperCase()}</div>}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1rem"}}>
        <div className="panel" style={{marginBottom:0}}><div className="panel-title" style={{marginBottom:".75rem"}}>Eligible Countries</div><div className="chips-row">{elig.eligible_countries?.length?elig.eligible_countries.map(c=><span key={c} className="chip chip-green">✓ {c}</span>):<span style={{color:"var(--text3)",fontSize:".85rem"}}>None matched</span>}</div></div>
        <div className="panel" style={{marginBottom:0}}><div className="panel-title" style={{marginBottom:".75rem"}}>Recommended Programs</div><div className="chips-row">{elig.recommended_programs?.map(p=><span key={p} className="chip chip-blue">{p}</span>)}</div></div>
      </div>
      {totalBudgetLKR>0&&<div className="panel" style={{marginTop:"1rem"}}><div className="panel-title" style={{marginBottom:".75rem"}}> Financial Summary</div><DataGrid pairs={[["Total Budget",formatLKR(totalBudgetLKR)],["Funding Source",fin.funding_source]]} /></div>}
      <div className="panel"><div className="panel-title" style={{marginBottom:".75rem"}}>English Proficiency</div>{engBadges.length?<div style={{display:"flex",gap:".5rem",flexWrap:"wrap"}}>{engBadges}</div>:<span style={{color:"var(--text3)",fontSize:".82rem",fontFamily:"var(--mono)"}}>No scores provided</span>}</div>
      {Array.isArray(elig.documents_considered)&&elig.documents_considered.length>0&&<div className="panel" style={{marginBottom:"1rem"}}><div className="panel-title" style={{marginBottom:".75rem"}}>Documents Considered</div><div className="chips-row">{elig.documents_considered.map((d)=><span key={d} className="chip chip-blue">{d}</span>)}</div></div>}
      {elig.notes?.length>0&&<div className="panel"><div className="panel-title" style={{marginBottom:".75rem"}}>Assessment Notes</div>{elig.notes.map((n,i)=><Alert key={i} type={elig.eligible?"ok":"warn"}>{n}</Alert>)}</div>}
      <div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Documents</button><button className="btn btn-primary" onClick={onNext}>View Universities →</button></div>
    </div>
  );
}

// ── UNIVERSITIES STEP ───────────────────────────────────────────────────────
const APP_STATUS_META = {
  submitted:    { label: "Submitted",    cls: "chip chip-blue" },
  under_review: { label: "Under Review", cls: "chip chip-amber" },
  accepted:     { label: "Accepted",     cls: "chip chip-green" },
  rejected:     { label: "Rejected",     cls: "chip chip-red" },
  withdrawn:    { label: "Withdrawn",    cls: "chip" },
};

function ApplicationsTracker({ user, refreshKey }) {
  const [apps, setApps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [busyId, setBusyId] = useState("");

  useEffect(() => {
    setLoading(true);
    fetchApplications(user?.token)
      .then(setApps)
      .catch(e => setErr(e.message || "Failed to load applications."))
      .finally(() => setLoading(false));
  }, [user?.token, refreshKey]);

  const doWithdraw = async (appId) => {
    if (!window.confirm("Withdraw this application?")) return;
    try {
      setBusyId(appId);
      await withdrawApplication(appId, user?.token);
      setApps(prev => prev.map(a => a.application_id === appId ? { ...a, status: "withdrawn" } : a));
    } catch (e) {
      setErr(e.message || "Withdraw failed.");
    } finally {
      setBusyId("");
    }
  };

  const active = apps.filter(a => a.status !== "withdrawn");
  const withdrawn = apps.filter(a => a.status === "withdrawn");

  if (loading) return <Alert type="info">Loading your applications…</Alert>;
  if (err) return <Alert type="error">{err}</Alert>;
  if (!apps.length) return <div style={{ fontFamily: "var(--mono)", fontSize: ".72rem", color: "var(--text3)", padding: ".75rem 0" }}>No applications submitted yet. Click "Apply Now" on a university below.</div>;

  return (
    <div>
      {active.map(app => {
        const sm = APP_STATUS_META[app.status] || APP_STATUS_META.submitted;
        const busy = busyId === app.application_id;
        return (
          <div key={app.application_id} style={{ padding: ".85rem 1rem", border: "1px solid var(--border)", borderRadius: "var(--r)", marginBottom: ".6rem", background: "var(--bg2)", display: "grid", gridTemplateColumns: "1fr auto", gap: ".75rem", alignItems: "start" }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: ".92rem" }}>{app.university_name}</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: ".68rem", color: "var(--text3)", marginTop: ".2rem" }}>{app.program} · {app.country}</div>
              {app.advisor_notes && <div style={{ fontFamily: "var(--mono)", fontSize: ".66rem", color: "var(--amber)", marginTop: ".3rem" }}>Advisor: {app.advisor_notes}</div>}
              <div style={{ fontFamily: "var(--mono)", fontSize: ".62rem", color: "var(--text3)", marginTop: ".2rem" }}>Submitted {new Date(app.submitted_at).toLocaleDateString()}</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: ".4rem" }}>
              <span className={sm.cls}>{sm.label}</span>
              {app.status === "submitted" && <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => doWithdraw(app.application_id)}>{busy ? "…" : "Withdraw"}</button>}
            </div>
          </div>
        );
      })}
      {withdrawn.length > 0 && (
        <div style={{ fontFamily: "var(--mono)", fontSize: ".65rem", color: "var(--text3)", marginTop: ".5rem" }}>
          {withdrawn.length} withdrawn application{withdrawn.length > 1 ? "s" : ""} not shown.
        </div>
      )}
    </div>
  );
}

function UniversitiesStep({profile, elig, onBack, onReset, user}){
  const fin=profile.financial||{};
  const budgetLKR=toLKR(parseFloat(fin.total_budget)||0,fin.budget_currency||"LKR");
  const [unis,setUnis]=useState([]);
  const [loading,setLoading]=useState(true);
  const [fetchErr,setFetchErr]=useState("");
  const [appliedKeys,setAppliedKeys]=useState(new Set());
  const [applyErr,setApplyErr]=useState("");
  const [applyBusy,setApplyBusy]=useState("");
  const [trackerKey,setTrackerKey]=useState(0);

  useEffect(()=>{
    setLoading(true);
    setFetchErr("");
    fetchUniversities(profile.country,elig.grade_point,profile.program_interest,budgetLKR,user?.token)
      .then(setUnis)
      .catch(e=>setFetchErr(e.message||"Failed to load universities."))
      .finally(()=>setLoading(false));
  },[profile.country,elig.grade_point,profile.program_interest,budgetLKR,user?.token]);

  // Pre-fill already-applied universities on mount
  useEffect(()=>{
    if(!user?.token) return;
    fetchApplications(user.token).then(apps=>{
      const keys=new Set(apps.filter(a=>a.status!=="withdrawn").map(a=>`${a.university_name}||${a.program}`));
      setAppliedKeys(keys);
    }).catch(()=>{});
  },[user?.token]);

  const handleApply = async (u, program) => {
    const key = `${u.name}||${program}`;
    if (appliedKeys.has(key)) return;
    setApplyErr("");
    setApplyBusy(key);
    try {
      await submitApplication({
        university_name: u.name,
        university_id: u.id || u.name.toLowerCase().replace(/\s+/g,"-"),
        program,
        country: u.country || profile.country || "",
        eligibility_tier: elig.eligibility_tier,
        grade_point: elig.grade_point,
        university_data: { qs: u.qs, location: u.location, website: u.website, tuitionDisplay: u.tuitionDisplay, minGpa: u.minGpa, ielts: u.ielts },
      }, user?.token);
      setAppliedKeys(prev => new Set([...prev, key]));
      setTrackerKey(k => k + 1);
    } catch(e) {
      if (e.message?.includes("Already applied")) {
        setAppliedKeys(prev => new Set([...prev, key]));
      } else {
        setApplyErr(e.message || "Application failed.");
      }
    } finally {
      setApplyBusy("");
    }
  };

  const affordable=unis.filter(u=>u.affordable),stretch=unis.filter(u=>!u.affordable);

  const UniCard=({u,idx})=>{
    const primaryProgram = u.programs[0] || profile.program_interest || "General";
    const key = `${u.name}||${primaryProgram}`;
    const applied = appliedKeys.has(key);
    const busy = applyBusy === key;
    return (
      <div className="uni-card">
        <div className="uni-header">
          <div className="uni-name">{idx}. {u.name}</div>
          <div style={{display:"flex",gap:".4rem",flexWrap:"wrap",justifyContent:"flex-end"}}>
            <div className="uni-qs">QS #{u.qs}</div>
            {u.affordable?<span className="fin-tag">✓ Within Budget</span>:<span className="fin-warn-tag">⚠ Stretch</span>}
          </div>
        </div>
        <div className="uni-meta"><span>{u.location}</span><span>{u.tuitionDisplay}</span></div>
        <div className="uni-tags">{u.programs.map(p=><span key={p} className="uni-tag">{p}</span>)}</div>
        <div className="uni-reqs">
          <div className="req-box"><div className="req-lbl">Min GPA</div><div className="req-val">{u.minGpa} / 4.0</div></div>
          <div className="req-box"><div className="req-lbl">IELTS Min</div><div className="req-val">{u.ielts}+</div></div>
        </div>
        <div style={{display:"flex",gap:".6rem",alignItems:"center",marginTop:".75rem",flexWrap:"wrap"}}>
          <a href={u.website} target="_blank" rel="noreferrer" className="uni-link" style={{flex:1}}>↗ Visit Website</a>
          {user?.token && (
            applied
              ? <span className="chip chip-green" style={{flexShrink:0}}>✓ Applied</span>
              : <button className="btn btn-primary btn-sm" style={{flexShrink:0}} disabled={busy} onClick={()=>handleApply(u, primaryProgram)}>
                  {busy ? "Submitting…" : "Apply Now"}
                </button>
          )}
        </div>
      </div>
    );
  };

  return(
    <div className="fade-up">
      {user?.token && (
        <div className="panel" style={{marginBottom:"1.5rem"}}>
          <div className="panel-title" style={{marginBottom:".75rem"}}>My Applications</div>
          <ApplicationsTracker user={user} refreshKey={trackerKey} />
        </div>
      )}
      {applyErr && <Alert type="error">{applyErr}</Alert>}
      <div style={{marginBottom:"1.25rem"}}>
        <div style={{fontFamily:"var(--sans)",fontSize:"1.05rem",fontWeight:800,marginBottom:".4rem"}}>Matched Universities — {profile.country}</div>
        {loading?<Alert type="info">Loading universities…</Alert>:fetchErr?<Alert type="error">{fetchErr}</Alert>:unis.length>0?<Alert type="ok">Found {unis.length} {unis.length===1?"university":"universities"} — {affordable.length} within budget</Alert>:<Alert type="warn">No universities matched. Adjust GPA, country, or program in your profile.</Alert>}
      </div>
      {!loading&&!fetchErr&&<>
        {affordable.length>0&&<div><div className="slabel">Within Budget</div>{affordable.map((u,i)=><UniCard key={u.name} u={u} idx={i+1} />)}</div>}
        {stretch.length>0&&<div><div className="slabel">Stretch — Consider Scholarship / Loan</div>{stretch.map((u,i)=><UniCard key={u.name} u={u} idx={affordable.length+i+1} />)}</div>}
      </>}
      <div className="btn-row"><button className="btn btn-ghost" onClick={onBack}>← Eligibility</button><button className="btn btn-primary" onClick={onReset}>New Search</button></div>
    </div>
  );
}

// ── AUTH ────────────────────────────────────────────────────────────────────
function AuthPage({onLogin}){
  const [tab,setTab]=useState("login");
  const [email,setEmail]=useState("");
  const [pw,setPw]=useState("");
  const [name,setName]=useState("");
  const [role,setRole]=useState("student");
  const [err,setErr]=useState("");
  const [ok,setOk]=useState("");
  const [loading,setLoading]=useState(false);

  const login=async()=>{
    const normalizedEmail = (email || "").trim();
    if(!normalizedEmail||!pw){
      setErr("Enter email and password.");
      return;
    }
    if(pw !== pw.trim()){
      setErr("Password contains leading or trailing spaces. Please remove extra spaces and try again.");
      return;
    }
    setErr("");
    setOk("");
    setLoading(true);
    try{
      const payload = await loginUser({ email: normalizedEmail, password: pw });
      const user = payload?.user || { email: normalizedEmail, name: normalizedEmail.split("@")[0] };
      onLogin({ ...user, token: payload?.token || "" });
    }catch(e){
      setErr(e.message || "Login failed.");
    }finally{
      setLoading(false);
    }
  };

  const register=async()=>{
    const normalizedName = (name || "").trim();
    const normalizedEmail = (email || "").trim();
    if(!normalizedName||!normalizedEmail||!pw){
      setErr("Fill all fields.");
      return;
    }
    if(pw !== pw.trim()){
      setErr("Password contains leading or trailing spaces. Please remove extra spaces and try again.");
      return;
    }
    setErr("");
    setOk("");
    setLoading(true);
    try{
      await registerUser({ name: normalizedName, email: normalizedEmail, password: pw, role });
      setOk("Account created. You can now sign in.");
      setTimeout(()=>{
        setOk("");
        setTab("login");
      },1200);
    }catch(e){
      setErr(e.message || "Register failed.");
    }finally{
      setLoading(false);
    }
  };

  return(
    <div className="auth-wrap">
      <div className="auth-card fade-up">
        <div className="auth-logo"><div className="auth-icon">🎓</div><span>UniAssist</span></div>
        
        <div className="tabs" style={{marginTop:"1.5rem"}}><button className={`tab${tab==="login"?" active":""}`} onClick={()=>{setTab("login");setErr("");}}>Login</button><button className={`tab${tab==="register"?" active":""}`} onClick={()=>{setTab("register");setErr("");}}>Register</button></div>
        {tab==="login" ? (
          <div>
            <div className="field" style={{marginBottom:"1rem"}}>
              <label className="flabel">Email</label>
              <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@email.com" onKeyDown={e=>e.key==="Enter"&&login()} />
            </div>
            <div className="field" style={{marginBottom:"1.5rem"}}>
              <label className="flabel">Password</label>
              <input type="password" value={pw} onChange={e=>setPw(e.target.value)} onKeyDown={e=>e.key==="Enter"&&login()} />
            </div>
            {err&&<Alert type="error">{err}</Alert>}
            {ok&&<Alert type="ok">{ok}</Alert>}
            <button className="btn btn-primary btn-full" onClick={login} style={{marginTop:"1rem"}} disabled={loading}>
              {loading ? "Signing In..." : "Sign In"}
            </button>
            <div style={{marginTop:".7rem", fontFamily:"var(--mono)", fontSize:".66rem", color:"var(--text3)", lineHeight:1.5}}>
              If you are using a fresh Colab runtime, local accounts may reset. Use <strong>Register</strong> first, then sign in.
            </div>
          </div>
        ) : (
          <div>
            <div className="field" style={{marginBottom:"1rem"}}>
              <label className="flabel">Full Name</label>
              <input value={name} onChange={e=>setName(e.target.value)} />
            </div>
            <div className="field" style={{marginBottom:"1rem"}}>
              <label className="flabel">Email</label>
              <input value={email} onChange={e=>setEmail(e.target.value)} />
            </div>
            <div className="field" style={{marginBottom:"1.5rem"}}>
              <label className="flabel">Password</label>
              <input type="password" value={pw} onChange={e=>setPw(e.target.value)} />
            </div>
            <div className="field" style={{marginBottom:"1.5rem"}}>
              <label className="flabel">Role</label>
              <select value={role} onChange={e=>setRole(e.target.value)}>
                <option value="student">Student</option>
                <option value="advisor">Advisor</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            {err&&<Alert type="error">{err}</Alert>}
            {ok&&<Alert type="ok">{ok}</Alert>}
            <button className="btn btn-primary btn-full" onClick={register} style={{marginTop:"1rem"}} disabled={loading}>
              {loading ? "Creating Account..." : "Create Account"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function UserCard({profile,user}){
  if(!profile.full_name)return null;
  const initials=profile.full_name.split(" ").slice(0,2).map(w=>w[0]).join("").toUpperCase();
  const tags=[profile.current_qualification,profile.country,profile.program_interest].filter(Boolean);
  return(<div className="user-card"><div className="user-avatar">{initials||"?"}</div><div style={{flex:1}}><div className="user-name">{profile.full_name}</div><div className="user-email">{user.email}{profile.district?` · ${profile.district}`:""}</div><div className="user-tags">{tags.map(t=><span key={t} className="user-tag">{t}</span>)}</div></div>{profile.financial?.total_budget&&<div style={{textAlign:"right",flexShrink:0}}><div className="fin-sum-label">Budget</div><div className="fin-sum-val" style={{fontSize:".82rem"}}>{formatLKR(toLKR(parseFloat(profile.financial.total_budget),profile.financial.budget_currency||"LKR"))}</div></div>}</div>);
}

// ── APP ROOT ────────────────────────────────────────────────────────────────
const STEPS=[{num:1,name:"Profile"},{num:2,name:"Documents"},{num:3,name:"Eligibility"},{num:4,name:"Universities"}];

export default function App(){
  const [user,setUser]=useState(null);
  const [step,setStep]=useState(1);
  const [profile,setProfile]=useState({});
  const [docData,setDocData]=useState({});
  const [elig,setElig]=useState(null);
  const [stateReady,setStateReady]=useState(false);
  const role = (user?.role || "student").toLowerCase();
  const dashboardLabel = role === "admin" ? "Admin Dashboard" : role === "advisor" ? "Advisor Dashboard" : "Student Dashboard";

  const handleDoc=(docPayload)=>{
    if(!docPayload) return;
    let normalized = docPayload;
    if(docPayload.document_type){
      normalized = {
        documents: {[docPayload.document_type]:docPayload},
        primary_document_type: docPayload.document_type,
        english_proficiency: docPayload.english_proficiency || {},
      };
    }
    if(!normalized.documents || !Object.keys(normalized.documents).length) return;
    const eligResult=assessEligibility(profile,normalized);
    setDocData(normalized);setElig(eligResult);setStep(3);
  };
  const reset=()=>{setStep(1);setProfile({});setDocData({});setElig(null);};
  const logout=()=>{setUser(null);setStateReady(false);reset();};

  useEffect(()=>{
    let cancelled = false;
    if(!user?.email){
      setStateReady(false);
      return ()=>{cancelled = true;};
    }
    (async()=>{
      try{
        const state = await fetchUserState(user.email, user?.token);
        if(cancelled) return;
        if(state && typeof state === "object"){
          if(typeof state.step === "number") setStep(state.step);
          if(state.profile && typeof state.profile === "object") setProfile(state.profile);
          if(state.docData && typeof state.docData === "object") setDocData(state.docData);
          if(state.elig && typeof state.elig === "object") setElig(state.elig);
        }
      }catch{
        // Continue with in-memory defaults if backend state is unavailable.
      }finally{
        if(!cancelled) setStateReady(true);
      }
    })();
    return ()=>{cancelled = true;};
  },[user?.email]);

  useEffect(()=>{
    if(!user?.email || !stateReady) return;
    const payload = { step, profile, docData, elig };
    saveUserState(user.email, payload, user?.token).catch(()=>{});
  },[user?.email,stateReady,step,profile,docData,elig]);

  if(!user) return <><style>{STYLES}</style><AuthPage onLogin={u=>{setUser(u);setProfile(p=>({...p,email:u.email}));}} /></>;

  if(role === "admin"){
    return (
      <>
        <style>{STYLES}</style>
        <div className="app">
          <div className="topbar">
            <div className="logo">
              <div className="logo-icon">🎓</div>
              <span>UniAssist</span>
            </div>
            <div style={{display:"flex",alignItems:"center",gap:"1rem"}}>
              <span style={{fontFamily:"var(--mono)",fontSize:".68rem",color:"var(--text3)"}}>{user.name||user.email}</span>
              <span className="user-tag" style={{textTransform:"uppercase"}}>{role}</span>
              <button className="btn btn-danger btn-sm" onClick={logout}>Logout</button>
            </div>
          </div>
          <div className="main">
            <AdminDashboard user={user} />
          </div>
        </div>
        <ChatBot user={user} />
      </>
    );
  }

  if(role === "advisor"){
    return (
      <>
        <style>{STYLES}</style>
        <div className="app">
          <div className="topbar">
            <div className="logo">
              <div className="logo-icon">🎓</div>
              <span>UniAssist</span>
            </div>
            <div style={{display:"flex",alignItems:"center",gap:"1rem"}}>
              <span style={{fontFamily:"var(--mono)",fontSize:".68rem",color:"var(--text3)"}}>{user.name||user.email}</span>
              <span className="user-tag" style={{textTransform:"uppercase"}}>{role}</span>
              <button className="btn btn-danger btn-sm" onClick={logout}>Logout</button>
            </div>
          </div>
          <div className="main">
            <AdvisorDashboard user={user} />
          </div>
        </div>
        <ChatBot user={user} />
      </>
    );
  }

  return(
    <>
      <style>{STYLES}</style>
      <div className="app">
        <div className="topbar">
          <div className="logo">
            <div className="logo-icon">🎓</div>
            <span>UniAssist</span>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:"1rem"}}>
            <span style={{fontFamily:"var(--mono)",fontSize:".68rem",color:"var(--text3)"}}>{user.name||user.email}</span>
            <span className="user-tag" style={{textTransform:"uppercase"}}>{role}</span>
            <button className="btn btn-danger btn-sm" onClick={logout}>Logout</button>
          </div>
        </div>
        <div className="main">
          <div className="hero">
            <div className="hero-eyebrow"><span className="hero-dot" />{dashboardLabel}</div>
            <h1>Find Your <em>Ideal</em><br />University Abroad</h1>
          </div>
          <div className="step-rail">
            {STEPS.map(s=>(<div key={s.num} className={`step-seg${step>s.num?" done":step===s.num?" active":""}`}><div className="step-circle">{step>s.num?"✓":s.num}</div><div className="step-name">{s.name}</div></div>))}
          </div>
          {step>1&&<UserCard profile={profile} user={user} />}
          {step===1&&<ProfileStep data={profile} onNext={d=>{setProfile(d);setStep(2);}} />}
          {step===2&&<DocumentStep profile={profile} docData={docData} onNext={handleDoc} onBack={()=>setStep(1)} user={user} />}
          {step===3&&elig&&<EligibilityStep elig={elig} docData={docData} profile={profile} onNext={()=>setStep(4)} onBack={()=>setStep(2)} />}
          {step===3&&!elig&&<div className="panel"><Alert type="error">Eligibility data missing. Go back and re-submit your document.</Alert><div className="btn-row btn-row-end" style={{marginTop:"1rem"}}><button className="btn btn-ghost" onClick={()=>setStep(2)}>← Back to Documents</button></div></div>}
          {step===4&&elig&&<UniversitiesStep profile={profile} elig={elig} onBack={()=>setStep(3)} onReset={reset} user={user} />}
        </div>
      </div>
      <ChatBot user={user} />
    </>
  );
}