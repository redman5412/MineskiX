// routes/overlayStyles.js — GET/POST /api/overlay-styles
const express  = require('express');
const router   = express.Router();
const fs       = require('fs');
const state    = require('../lib/state');
const projects = require('../lib/projects');

function load() {
  try { return JSON.parse(fs.readFileSync(projects.getStylesFilePath(), 'utf8')); }
  catch (e) { return {}; }
}

router.get('/api/overlay-styles', function (req, res) {
  var file   = req.query.file || '';
  var styles = load();
  res.json(styles[file] || {});
});

router.post('/api/overlay-styles', function (req, res) {
  var body   = req.body || {};
  var file   = body.file;
  var styles = body.styles;
  if (!file || !styles) return res.status(400).json({ error: 'missing file or styles' });
  var all    = load();
  all[file]  = styles;
  fs.writeFileSync(projects.getStylesFilePath(), JSON.stringify(all, null, 2));
  /* Force every open overlay scene (Fullscreen.html, ingame.html,
     Draft.html, DraftIndex.html, ENTVC.html — anywhere the html is
     loaded) to hard-reload so this save's position/size overrides take
     effect immediately, instead of waiting for a manual refresh. Safe
     to do unattended: edits are never made while a scene is live. */
  state.overlayClients.forEach(function (c) { try { c.write('event: reload\ndata: {}\n\n'); } catch (e) {} });
  res.json({ ok: true });
});

// POST /api/overlay-nudge — targeted single-selector position patch used by
// the dashboard Control tab's quick nudge buttons (MVP Scene/Highlights
// photo+hero repositioning). Persists the same way the Edit tab's Save
// does, but broadcasts a lightweight 'stylepatch' event instead of
// 'reload' — Fullscreen.html applies it live via CSS injection with no page
// reload, since the primary feedback loop here is the dashboard's own
// live preview iframe and a reload-triggered flash on every click would
// defeat the point of a quick nudge button.
router.post('/api/overlay-nudge', function (req, res) {
  var body     = req.body || {};
  var file     = body.file;
  var selector = body.selector;
  var props    = body.props;
  if (!file || !selector || !props) return res.status(400).json({ error: 'missing file, selector, or props' });
  var all = load();
  if (!all[file]) all[file] = {};
  if (!all[file][selector]) all[file][selector] = {};
  Object.assign(all[file][selector], props);
  fs.writeFileSync(projects.getStylesFilePath(), JSON.stringify(all, null, 2));
  var payload = JSON.stringify({ file: file, selector: selector, props: props });
  state.overlayClients.forEach(function (c) { try { c.write('event: stylepatch\ndata: ' + payload + '\n\n'); } catch (e) {} });
  res.json({ ok: true });
});

// GET /api/fonts — every font file under fonts/ and projects/redbull/fonts/,
// grouped by family → styles, for the Edit tab's font picker (Red Bull configs).
// Scanned fresh on each call so a font dropped into either folder shows up
// without a restart. Italic/variable files are skipped; when the same face
// exists in several formats the smallest web format wins (woff2 > woff > otf > ttf).
var FONT_DIRS = ['fonts', 'projects/redbull/fonts'];
var FONT_EXT_RANK = { '.woff2': 0, '.woff': 1, '.otf': 2, '.ttf': 3 };
var FONT_STYLE_WEIGHTS = [   // longest names first so 'extrabold' wins over 'bold'
  ['extralight', 200], ['semibold', 600], ['extrabold', 800], ['condbold', 700],
  ['regular', 400], ['medium', 500], ['heavy2', 900], ['heavy1', 900], ['heavy', 900],
  ['black', 900], ['light', 300], ['thin2', 100], ['thin1', 100], ['thin', 100], ['book', 400], ['bold', 700],
];
function prettyFamily(s) { return s.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^Futurafor /, 'Futura for '); }
function parseFontName(base) {
  base = base.replace(/-BF[0-9a-f]{6,}$/i, '');              // strip download hash suffixes
  if (/italic|variable|var$/i.test(base)) return null;
  var m = base.match(/^(.*?)[-_ ]?([A-Za-z0-9]+)$/);
  var lower = base.toLowerCase();
  for (var i = 0; i < FONT_STYLE_WEIGHTS.length; i++) {
    var s = FONT_STYLE_WEIGHTS[i][0];
    if (lower.length > s.length && lower.slice(-s.length) === s) {
      var fam = base.slice(0, -s.length).replace(/[-_ ]+$/, '');
      var style = base.slice(-s.length);
      if (s === 'condbold') { fam += ' Cond'; style = 'Bold'; }
      return { family: prettyFamily(fam), style: style.charAt(0).toUpperCase() + style.slice(1), weight: FONT_STYLE_WEIGHTS[i][1] };
    }
  }
  if (!m) return null;
  return { family: prettyFamily(base), style: 'Regular', weight: 400 };
}
router.get('/api/fonts', function (req, res) {
  var path = require('path');
  var root = path.join(__dirname, '..');
  var faces = {};   // "family|style" -> { family, style, weight, url, rank }
  function walk(dir) {
    var entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return; }
    entries.forEach(function (e) {
      var full = path.join(dir, e.name);
      if (e.isDirectory()) return walk(full);
      var ext = path.extname(e.name).toLowerCase();
      if (!(ext in FONT_EXT_RANK)) return;
      var p = parseFontName(path.basename(e.name, path.extname(e.name)));
      if (!p) return;
      var key = p.family + '|' + p.style;
      var url = '/' + path.relative(root, full).split(path.sep).map(encodeURIComponent).join('/');
      if (!faces[key] || FONT_EXT_RANK[ext] < faces[key].rank) {
        faces[key] = { family: p.family, style: p.style, weight: p.weight, url: url, rank: FONT_EXT_RANK[ext] };
      }
    });
  }
  FONT_DIRS.forEach(function (d) { walk(path.join(root, d)); });
  var families = {};
  Object.keys(faces).forEach(function (k) {
    var f = faces[k];
    (families[f.family] = families[f.family] || []).push({ style: f.style, weight: f.weight, url: f.url });
  });
  var out = Object.keys(families).sort().map(function (name) {
    return { family: name, styles: families[name].sort(function (a, b) { return a.weight - b.weight || a.style.localeCompare(b.style); }) };
  });
  res.set('Cache-Control', 'no-store').json(out);
});

module.exports = router;
