"""Restrained document-workspace styling for the four user-facing pages."""

CLOUD_UI_CSS = """
:root { --smw-ink:#172638; --smw-muted:#405167; --smw-line:#bac7d4;
  --smw-lavender:#674d9e; --smw-mint:#e8f5ee; --smw-blue:#eaf3fb;
  --smw-gold:#fff5db; --smw-silver:#f6f8fa; }
.gradio-container, .gradio-container.dark, .dark .gradio-container {
  color-scheme:light !important;
  --body-text-color:#172638 !important;
  --body-text-color-subdued:#405167 !important;
  --background-fill-primary:#fff !important;
  --background-fill-secondary:#f1f4fa !important;
  --block-label-text-color:#27384d !important;
  --block-title-text-color:#172638 !important;
  --block-info-text-color:#405167 !important;
  --input-background-fill:#fff !important;
  --input-placeholder-color:#4c5e73 !important;
  --input-border-color:#aebdcc !important;
  --button-secondary-background-fill:#f2f5f9 !important;
  --button-secondary-background-fill-hover:#e8ecf4 !important;
  --button-secondary-text-color:#27384d !important;
  --button-secondary-text-color-hover:#172638 !important;
  --checkbox-label-text-color:#27384d !important;
  --checkbox-label-text-color-selected:#402b78 !important;
  --checkbox-border-color:#8291a5 !important;
  --link-text-color:#4e348e !important;
  --border-color-primary:#bac7d4 !important;
  --block-border-color:#bac7d4 !important;
}
body, .gradio-container { background:#fafbfd !important; color:var(--smw-ink) !important; }
.gradio-container .block:not(.smw-header), .gradio-container .tabitem,
.gradio-container .form, .gradio-container .wrap, .gradio-container .input-container {
  background:#fff !important; color:var(--smw-ink) !important;
  border-color:var(--smw-line) !important; }
.gradio-container label, .gradio-container .label-wrap, .gradio-container .info {
  color:var(--smw-ink) !important; background:transparent !important; opacity:1 !important; }
.gradio-container [data-testid="block-info"] { color:#405167 !important; opacity:1 !important; }
.gradio-container input::placeholder,.gradio-container textarea::placeholder {
  color:#4c5e73 !important; opacity:1 !important; }
.gradio-container .upload-container,.gradio-container .file-preview,
.gradio-container .accordion summary { color:#27384d !important; }
.gradio-container .upload-container svg,.gradio-container .file-preview svg,
.gradio-container .label-wrap svg,.gradio-container .accordion svg,
.gradio-container button:not(:disabled):not(.primary) svg {
  color:#405167 !important; opacity:1 !important; }
.gradio-container [role="tab"] { color:#435265 !important; background:transparent !important;
  border-radius:8px 8px 0 0 !important; }
.gradio-container [role="tab"][aria-selected="true"] { color:#493382 !important;
  border-color:#745db2 !important; background:#f1edfa !important; }
.gradio-container .tab-nav, .gradio-container .tabs { background:transparent !important; }
.gradio-container { max-width:1500px !important; padding:20px 24px 42px !important;
  font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif !important; letter-spacing:0 !important; }
.gradio-container h1 { font-size:25px !important; line-height:1.25 !important; margin:0 0 4px !important;
  color:#293145 !important; }
.gradio-container h2,.gradio-container h3 { letter-spacing:0 !important; }
.gradio-container .tabs { border-color:var(--smw-line) !important; }
.gradio-container .tab-nav { border-bottom:1px solid var(--smw-line) !important;
  gap:8px !important; padding:0 0 6px !important; }
.gradio-container .tab-nav button { border-radius:8px !important; color:#435265 !important;
  background:#fff !important; min-height:42px !important; padding:8px 15px !important; }
.gradio-container .tab-nav button.selected { background:#eeeaf8 !important;
  color:#493382 !important; border-color:#c8b9e7 !important; font-weight:650 !important; }
.gradio-container input,.gradio-container textarea,.gradio-container select,
.gradio-container button,.gradio-container .wrap { border-radius:12px !important; }
.gradio-container input,.gradio-container textarea,.gradio-container select {
  border-color:#cfd8e2 !important; background:#fff !important; color:var(--smw-ink) !important; }
/* Dropdown options may be rendered in a fixed overlay outside the field wrapper. */
body ul.options[role="listbox"], .gradio-container [role="listbox"],
.gradio-container .dropdown .options {
  color-scheme:light !important; background:#fff !important;
  color:#172638 !important; border:1px solid #bac7d4 !important;
  border-radius:8px !important; box-shadow:0 8px 24px #17263822 !important; }
body ul.options[role="listbox"] li[role="option"], .gradio-container [role="option"],
.gradio-container .dropdown .options .item {
  background:#fff !important; color:#172638 !important; opacity:1 !important; }
body ul.options[role="listbox"] li[role="option"]:hover,
body ul.options[role="listbox"] li[role="option"][aria-selected="true"],
body ul.options[role="listbox"] li[role="option"].active,
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"],
.gradio-container .dropdown .options .item:hover {
  background:#eeeaf8 !important; color:#402b78 !important; }
body ul.options[role="listbox"] li[role="option"] .inner-item {
  color:inherit !important; }
.gradio-container select option { background:#fff !important; color:#172638 !important; }
.gradio-container input:focus,.gradio-container textarea:focus,.gradio-container select:focus,
.gradio-container button:focus-visible { outline:2px solid #745db2 !important; outline-offset:2px !important; }
.gradio-container button.primary { background:#745db2 !important; border-color:#745db2 !important;
  color:#fff !important; font-weight:650 !important; }
.gradio-container button.primary:hover { background:#604b9b !important; }
.gradio-container .accordion { border:1px solid var(--smw-line) !important;
  border-radius:8px !important; background:var(--smw-silver) !important; }
.smw-header { padding:12px 0 16px; border-bottom:1px solid var(--smw-line); margin-bottom:18px; }
.smw-header p { color:var(--smw-muted); margin:4px 0 0; }
.smw-section-note { color:var(--smw-muted); font-size:13px; margin:0 0 16px; }
.smw-view { background:#fafbfd !important; border-color:var(--smw-line) !important; border-radius:8px; }
.smw-panel,.smw-stat,.smw-comparison-section { box-shadow:none !important;
  border-color:var(--smw-line) !important; border-radius:8px !important; }
.smw-stage.success { border-top-color:#49966d; }.smw-stage.warning { border-top-color:#c49b44; }
.smw-stage.running { border-top-color:#5f9dc8; }
.smw-table th { background:var(--smw-blue) !important; }
.smw-diff-summary,.smw-interpretation-overview { background:var(--smw-blue) !important; }
.smw-warning { background:var(--smw-gold) !important; }
.smw-plain-minutes { max-width:980px; margin:0 auto; padding:28px 32px;
  background:#fff; border:1px solid var(--smw-line); border-radius:8px; }
.smw-plain-minutes pre { white-space:pre-wrap; overflow-wrap:anywhere;
  font:400 15px/1.8 Inter,"PingFang SC","Microsoft YaHei",sans-serif; color:var(--smw-ink); }
.gradio-container footer { display:none !important; }
.smw-appbar{align-items:center!important;gap:14px!important;padding:4px 2px 14px!important;border-bottom:1px solid var(--smw-line);margin-bottom:4px!important}
.smw-appbar>div{min-width:0}
.smw-appbar .block{border:0!important;box-shadow:none!important}
.smw-brand{display:flex;align-items:center;gap:9px;white-space:nowrap;color:#26334b;font-size:18px}
.smw-brand strong{color:#26334b!important;opacity:1!important}
.smw-brand-mark{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:7px;background:#7962b3;color:#fff;font-weight:750;font-size:17px}
.smw-model-chip{display:flex;align-items:center;gap:7px;white-space:nowrap;color:#526175;font-size:12px}
.smw-model-chip span{width:7px;height:7px;border-radius:50%;background:#9ba8b7}
.smw-primary-tabs>.tab-nav{margin-bottom:10px!important}
.smw-page-title{margin:0 0 12px!important}
.smw-page-title h2{font-size:23px!important;font-weight:700!important;margin:0!important;color:#26334b!important}
.smw-page-columns,.smw-compare-layout,.smw-results-layout{gap:18px!important;align-items:stretch!important}
.smw-side-form,.smw-compare-source{border-left:1px solid #e4e9ef;padding-left:18px!important}
.smw-field-caption,.smw-section-heading{font-weight:700;color:#25334b;font-size:14px;margin:5px 0 10px}
.smw-section-heading{border-top:1px solid var(--smw-line);padding-top:14px;margin-top:18px}
.smw-mode-tabs .tab-nav{display:flex;gap:0!important;border:0!important;background:#f0f3f7!important;border-radius:8px!important;padding:3px!important}
.smw-mode-tabs .tab-nav button{flex:1!important;min-height:38px!important;text-align:center!important;border:0!important;background:transparent!important;border-radius:6px!important}
.smw-mode-tabs .tab-nav button.selected{background:#e7e1f8!important;color:#514084!important}
.smw-upload{min-height:120px}
.smw-file-dates{margin-top:8px}
.smw-file-dates table th,.smw-file-dates [role="columnheader"]{background:#eaf3fb!important;color:#2b4f69!important;font-weight:700!important}
.smw-file-dates table td{background:#fff!important;color:#2c3c50!important}
.smw-file-dates button[aria-label="Add row"]{display:none!important}
.smw-upload .upload-container,.smw-upload .file-preview{max-height:145px!important;min-height:110px!important}
.smw-ready{background:#e7f5ee;color:#2b7754;border-radius:6px;padding:10px 13px;font-weight:650;font-size:13px;margin:4px 0}
.smw-ready.muted{background:#f1f4f7;color:#667382}
.smw-ready span{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;background:#4b9970;color:#fff;margin-right:4px;font-size:11px}
.smw-scope-chip{display:inline-block;padding:7px 12px;background:#e9f3f8;color:#386b84;border-radius:6px;font-size:13px;margin-bottom:8px}
.smw-history-available{border:1px solid #cfe7db;background:#edf7f1;color:#2d7354;border-radius:6px;padding:9px 11px;font-size:13px}
.smw-history-available span{display:block;color:#577264;font-size:11px;margin-top:2px}
.smw-progress{display:flex;gap:12px;align-items:center;border-radius:6px;border:1px solid #d9e8f2;background:#eef6fb;color:#365f79;padding:9px 11px;font-size:12px}
.smw-progress strong{white-space:nowrap;font-size:13px}
.smw-progress span{overflow-wrap:anywhere}
.smw-progress-completed{background:#eaf6ee;border-color:#cfe8da;color:#2d7351}
.smw-progress-completed_with_warnings{background:#fff7e4;border-color:#f0dfb2;color:#845f21}
.smw-progress-failed{background:#fff1ed;border-color:#efd0c8;color:#984938}
.smw-segmented .wrap{display:flex!important;gap:3px!important;flex-wrap:wrap!important;background:#f0f3f7!important;padding:3px!important;border-radius:7px!important}
.smw-segmented .wrap label{flex:1;min-width:100px;text-align:center;border-radius:5px!important;margin:0!important;padding:5px 10px!important;color:#27384d!important;opacity:1!important}
.smw-segmented .wrap label:has(input:checked){background:#e4def8!important;color:#402b78!important;border-color:#b8a6de!important;font-weight:700}
.smw-compare-top{align-items:end!important;max-width:370px}
.smw-compare-layout{border-top:1px solid var(--smw-line);padding-top:15px;margin-top:9px}
.smw-candidate-row{gap:10px!important}
.smw-candidate{border:1px solid #e1e7ee!important;border-radius:7px!important;background:#fff!important}
.smw-candidate .label-wrap{font-weight:700}
.smw-result-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:15px;padding:0 2px 11px;border-bottom:1px solid var(--smw-line);margin-bottom:8px}
.smw-result-heading h2{margin:0!important;font-size:20px!important;color:#26334b!important}
.smw-result-heading p{font-size:12px;color:#6d7887;margin:4px 0 0}
.smw-result-actions{justify-content:flex-end!important;gap:7px!important;margin-bottom:6px}
.smw-result-actions button{max-width:145px;min-height:36px!important}
.smw-download-links{gap:8px!important}
.smw-download-links>.block{max-width:260px}
.smw-run-list{display:grid;gap:6px;max-height:440px;overflow-y:auto;padding-right:2px}
.smw-run-selector .wrap{display:grid!important;gap:6px!important;max-height:450px;overflow-y:auto;padding-right:2px}
.smw-run-selector .wrap label{display:flex!important;align-items:flex-start!important;border:1px solid #bac7d4!important;border-radius:7px!important;padding:9px 10px!important;background:#fff!important;color:#27384d!important;opacity:1!important;font-size:12px!important;line-height:1.5!important}
.smw-run-selector .wrap label:has(input:checked){background:#eeeaf8!important;border-color:#bfb0df!important;color:#463574!important;font-weight:700!important}
.smw-run-item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 6px;border:1px solid #e0e6ed;border-radius:7px;padding:10px;background:#fff;min-width:0}
.smw-run-item strong{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#2a3c57}
.smw-run-item small{grid-column:1 / -1;color:#405167;font-size:11px}
.smw-status{display:inline-flex;align-items:center;white-space:nowrap;background:#eef2f6;color:#5a6776;border-radius:5px;padding:3px 7px;font-size:11px;font-weight:700}
.smw-status-completed{background:#e2f4e9;color:#2c7952}
.smw-status-completed_with_warnings{background:#fff1d5;color:#775017}
.smw-status-failed{background:#ffe8e4;color:#963d31}
.smw-status-running,.smw-status-queued{background:#e7f1fb;color:#3b739c}
.smw-list-empty{border:1px dashed #bac7d4;background:#f8fafc;color:#405167;padding:15px;border-radius:7px;font-size:13px}
@media(max-width:760px) {
  .gradio-container { padding:12px 10px 28px !important; }
  .gradio-container .tab-nav { overflow-x:auto; flex-wrap:nowrap; }
  .gradio-container .tab-nav button { white-space:nowrap; }
  .smw-scroll { overflow-x:auto; }
  .smw-appbar{flex-wrap:wrap!important}
  .smw-appbar>div{flex:1 1 150px}
  .smw-side-form,.smw-compare-source{border-left:0;border-top:1px solid #e4e9ef;padding-left:0!important;padding-top:12px!important}
  .smw-result-actions{justify-content:stretch!important}
  .smw-result-actions button{flex:1;max-width:none}
  .smw-run-list{max-height:240px}
}
"""
