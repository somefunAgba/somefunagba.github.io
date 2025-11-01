 
/*Note helpers*/
// const notes = document.querySelectorAll('.note');
// const track = document.querySelector('.note-track');

// references
const track = document.getElementById('note-track');
const notes = Array.from(track.querySelectorAll('.note'));
const progressEl = document.getElementById('note-progress');
const nextBtn = document.getElementById('next');
const prevBtn = document.getElementById('prev');

let current = 0;
// const widthPx = track.querySelector('.note').getBoundingClientRect().width;

// Update UI to a given index
async function showNote(index, {animate = true} = {}) {

  // optional, control transition
  track.style.transitionDuration = animate ? '500ms' : '0ms';

  // process index to update global current index
  index = Math.max(0, index);// Math.min(index, notes.length - 1));
  current = (index % notes.length);

  track.style.transform = `translateX(-${current * 100}%)`;
  // track.style.transform = `translateX(-${current * widthPx}px)`;

  // Update  progress
  progressEl.textContent = `${current + 1} / ${notes.length}`;

  // Save to localStorage
  localStorage.setItem('noteIndex', current);

  // Dispatch a custom event for plot sync
  const evt = new CustomEvent('notechange', { detail: { index: current } });
  document.dispatchEvent(evt);
}

// Controls
nextBtn.addEventListener('click', () => showNote(current + 1));
prevBtn.addEventListener('click', () => showNote(current - 1));

// Keyboard navigation
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') showNote(current + 1);
  if (e.key === 'ArrowLeft') showNote(current - 1);
});

// Basic swipe (touch)
let startX = null;
track.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; }, { passive: true });
track.addEventListener('touchmove', (e) => {}, { passive: true });
track.addEventListener('touchend', (e) => {
  if (startX === null) return;
  const dx = e.changedTouches[0].clientX - startX;
  if (dx < -30) showNote(current + 1);
  else if (dx > 30) showNote(current - 1);
  startX = null;
});

// Resize handling (if container width changes)
window.addEventListener('resize', () => showNote(current, { animate: false }));

// Example: sync with a plot (listen for external events)
// document.addEventListener('plotstep', (e) => {
//   // e.detail.step could be 0..notes.length-1
//   showNote(e.detail.step);
// });

// Initialize
// showNote(0, { animate: false });

// Load saved index or default to 0
// Now when the user reloads the page, the viewer will automatically return to the last note they were viewing. This works across sessions as long as localStorage is available.
const savedIndex = parseInt(localStorage.getItem('noteIndex'), 10);
showNote(Number.isFinite(savedIndex) ? savedIndex : 0, { animate: false });





/* Model helpers */
function etafcn(beta, gamma){ 
  let eta = 1
  if (!israw){
      eta = (1 - beta) / (1 - gamma)
  } 
  return eta; 
}

function hbfcn(beta){ 
    return 0*beta; 
}

function nagfcn(beta){ 
    return beta/(1+beta); 
}

function sfunfcn(beta){ 
    return 1 - Math.sqrt(2*(1-beta)); 
}
function sysPole(alpha, beta, gamma, eigval){ 
    return beta - etafcn(beta,gamma) * alpha * eigval; 
}

function lrstablerng(beta, gamma, eigval, nonnegative=true){
  const eta = etafcn(beta,gamma);
  let aMin = 1e-10;
  let aMaxlp = beta / (eta * eigval);
  if (israw){
    aMaxlp = 0.33 / eigval;
    // aMax = 1/eigval;
  }
  let aMax = (1+beta)/(eta*eigval);
  const aMaxhpSafe = 0.99*aMax;
  if(Math.abs(aMax - aMin) < 1e-8) aMax = aMin + 1.0;
  return {aMin,aMax,eta, aMaxlp, aMaxhpSafe};
}

/* Discrete-time simulation of change-level model:
   s_{t+1} = a*s_t - b * x_{t-1}
   x_t = x_{t-1} + s_t
   where a = beta - eta*alpha*lam, b = alpha*(1-beta)*lam
   We simulate for T steps given input u_t added to x (e.g., step or impulse).
*/
// Generate one Gaussian random number (mean 0, std 1)
function randn_bm() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); // avoid 0
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
  
// Add Gaussian noise with given mean and std
function addGaussianNoise(mean = 0, std = 0.01) {
    return randn_bm() * std + mean;
}
  
function simulateResponse(beta,gamma,lam,alpha, input='step', T=220){
  const eta = etafcn(beta,gamma);
  const eal = eta*alpha*lam;
  let a = 0, b = 0;
  if (israw){
    a = -eal;
    b = -eal;
  } else {
    a = beta - eal;
    b = -eal*(1 - gamma);
  }

  let s_prev = 0.0;
  let w_star = 1;
  let x_prev = -w_star;
  const xs = [];
  const ss = [];
  for(let t=0;t<T;t++){
    // input to x_{t-1} (we model external desired reference r_t; system tries to track; here use r constant)
    let r_prev = 0.0;
    if(input === 'step'){
      r_prev = 1.0; // step of magnitude 1 applied at time 0 (so x sees it from t=0)
    } else if(input === 'impulse'){
      x_prev = (t===0) ? -w_star : 0.0;
    }
    // For this simplified error dynamics, consider x_t is system output accumulating s_t and r is external reference added to x.
    // Equivalent discrete update:
    // s_t = a*s_{t-1} - b * x_{t-1}
    const n_t = 0; // addGaussianNoise(std=0)
    const s_t = a * s_prev + b * (x_prev + n_t); // add a noise-term to x_prev 
    const x_t = s_t + x_prev; // include reference directly (so error w.r.t 0)
    xs.push(x_t);
    ss.push(s_t);
    s_prev = s_t;
    x_prev = x_t;
  }
  return {xs, ss, a, b};
}


/* DOM refs */
const betaEl = document.getElementById('beta'), 
gammaEl = document.getElementById('gamma'),
lamEl = document.getElementById('eigval'), 
alphaEl = document.getElementById('alpha'), 
inputEl = document.getElementById('inputType');

const betaVal = document.getElementById('betaVal'), 
gammaVal = document.getElementById('gammaVal'),
lamVal = document.getElementById('lamVal'), 
alphaVal = document.getElementById('alphaVal');

const stabilityEl = document.getElementById('stability');
const playBtn = document.getElementById('play'), 
pauseBtn = document.getElementById('pause'), 
resetBtn = document.getElementById('reset');

const freeBtn = document.getElementById('free');
const phbBtn = document.getElementById('PHB'), 
nagBtn = document.getElementById('NAG'), 
sfunBtn = document.getElementById('SFUN'),
optBtn = document.getElementById('OPT'),
rawBtn = document.getElementById('NOF');
// Flag: is gamma locked to beta?
let gammafree= true;
let israw = false;
rawBtn.disabled = false;
freeBtn.disabled = false;

const poleValEl = document.getElementById('poleVal');
const gainValEl = document.getElementById('ingainVal');
const alphaLimitsEl = document.getElementById('alphaLimits');
const poleLimitsEl = document.getElementById('poleLimits');

const traceAValEl = document.getElementById('traceAVal');
const detAValEl = document.getElementById('detAVal');

const fpass = document.querySelectorAll('#fstate');

let params = {beta:parseFloat(betaEl.value), 
    gamma:parseFloat(gammaEl.value), 
    eigval:parseFloat(lamEl.value), 
    alpha:parseFloat(alphaEl.value), 
    input: inputEl.value};


/* Plot layout helpers */
const unitTheta = Array.from({length:401},(_,i)=>i*2*Math.PI/400);
const unitX = unitTheta.map(t=>Math.cos(t)), unitY = unitTheta.map(t=>Math.sin(t));

function overlayDataOneState(beta,gamma,eigval){
  const {aMin,aMax,eta,aMaxlp,aMaxhpSafe} = lrstablerng(beta,gamma,eigval,true);
  const n = 360;
  const alphas = Array.from({length:n},(_,i)=>aMin + (aMaxhpSafe-aMin)*i/(n-1));
  const poles = alphas.map(a => sysPole(a,beta,gamma,eigval));
  // Collect all values above threshold into one list
  const alphashp = alphas.filter(val => val > aMaxlp);
  const poleshp = alphashp.map(a => sysPole(a,beta,gamma,eigval));
  const z = Array(n).fill(0);
  return {alphas,poles,z,aMin,aMax,eta,aMaxlp,aMaxhpSafe, alphashp, poleshp};
}

/* ===== Error plot (time-domain) ===== */
function drawSys(){
  // simulate
  const T = 120;
  const sim = simulateResponse(params.beta, params.gamma, params.eigval, params.alpha, params.input, T);
  const gain = sim.b;
  gainValEl.textContent = `${gain.toFixed(3)}`;
  poleValEl.textContent = `${sim.a.toFixed(3)}`;

  const xs = sim.xs;
  const ss = sim.ss;
  const t = Array.from({length:xs.length}, (_,i)=>i);
  const peak = Math.max(...xs.map(Math.abs));
  const ylim = Math.max(1.0, peak*1.2);

  const traceX = {x:t, y:xs, mode:'lines', line:{color:'chocolate', width:1}, name:'$\\hbox{error, } ε[t]$'};
  const traceS = {x:t, y:ss, mode:'lines', line:{color:'coral', dash:'dot', width: 1}, name:'$\\hbox{change, } \\Delta ε[t]$'};
  const layout = {
    title: { 
      text: `$\\text{iteration-domain (${params.input})}$`,
      font: { size: 12, family: 'Arial, sans-serif', color: 'black' } 
    },
    xaxis: { title: { text: '$t$' }, range: [0, t.length - 1] },
    yaxis: { title: { text: '$\\hbox{signals}$' }, range: [-ylim, ylim] },
    // width: 300,
    // height: 300,
    showlegend: true,
    autosize: true,
  };
  
  Plotly.react('leftPlotc',[traceX, traceS], layout, {displayModeBar:false, responsive:true});
  
}
    

function chareq(a1, a0) {
    // Compute discriminant
    const disc = a1 * a1 - 4 * a0;
    let roots;
  
    if (disc >= 0) {
      // Real roots
      const r1 = (a1 + Math.sqrt(disc)) / 2;
      const r2 = (a1 - Math.sqrt(disc)) / 2;
      roots = [ [r1, 0], [r2, 0] ];
    } else {
      // Complex roots
      const real = a1 / 2;
      const imag = Math.sqrt(-disc) / 2;
      roots = [ [real, imag], [real, -imag] ];
    }
    return roots;
}

function getroots(gamma, eta, eigval, alphas, poles) {

    const a1s = poles.map(p => 1 + p);
    const a0s = poles.map((p,i) => p + eta * alphas[i] * eigval * (1 - gamma));

    // Arrays to hold traces of each root separately
    const root1re = [];
    const root1im = [];
    const root2re = [];
    const root2im = [];

    a0s.forEach((a0, i) => {
        const [r1, r2] = chareq(a1s[i], a0);
        root1re.push(r1[0]);
        root1im.push(r1[1]);
        root2re.push(r2[0]);
        root2im.push(r2[1]);
    });

    return {
        root1pts: { re: root1re, im: root1im },
        root2pts: { re: root2re, im: root2im }
    };
}

/* Right (interactive) plot update */
function drawRight(){
  const pole = sysPole(params.alpha, params.beta, params.gamma, params.eigval);
  
  const stable = Math.abs(pole) < 1.0;
  const highpass = pole < 0;

  fpass.forEach((fp) => {
    fp.textContent = highpass ? 'Highpass' : 'Lowpass';
  });
  
  stabilityEl.textContent = highpass ? 'Highpass' : 'Lowpass';
  stabilityEl.style.color = highpass ? 'crimson' : 'yellowgreen';
  betaVal.textContent = params.beta.toFixed(2);
  gammaVal.textContent = params.gamma.toFixed(2);
  lamVal.textContent = params.eigval.toFixed(2);
  alphaVal.textContent = params.alpha.toFixed(3);

  // recompute limits
  const {alphas,poles,z,aMin,aMax,eta,aMaxlp,aMaxhpSafe, alphashp, poleshp} = overlayDataOneState(params.beta, params.gamma, params.eigval);
  alphaEl.min = aMin; alphaEl.max = aMaxhpSafe;
  const nlenp = poles.length;
  const nlenph = poleshp.length;
  const zerosp = Array(nlenp).fill(0); 
  const zerosph = Array(nlenph).fill(0); 
  const polemax = sysPole(aMin, params.beta, params.gamma, params.eigval);
  const polemin = sysPole(aMaxhpSafe, params.beta, params.gamma, params.eigval);
  if (!israw){
    const eal = eta*params.alpha*params.eigval;
    gammaEl.min = 0.9*(params.beta-1)/(eal);
    gammaEl.max = params.beta;
  }

  // Update info panel
  // recompute α bounds and clamp alpha slider range/position
  alphaLimitsEl.textContent = `(0, ${aMaxlp.toFixed(3)})`;
  if (!israw){
    poleLimitsEl.textContent = `(0, ${params.beta.toFixed(3)})`;
  } else {
    poleLimitsEl.textContent = `(-0.5, 0)`;
  }
  
  if (highpass) {
    poleValEl.style.color = "crimson";
  } else {
    poleValEl.style.color = "yellowgreen";
  }

//   const ingain = -eta*params.alpha*params.eigval*(1-params.gamma);
  let a1 = 1 + pole  // trace
  let a0 = 0;
  if (!israw){
    //  a0 = pole - ingain // det
    a0 = params.beta - eta*params.alpha*params.eigval*params.gamma;
  } 
  
  const jury1 = (1+a1+a0)>0
  const jury2 = (1-a1+a0)>0
  const jury3 = Math.abs(a0)<1
  jstable = jury1 && jury2 && jury3
  roots = chareq(a1, a0);

  const {root1pts, root2pts} = getroots(params.gamma, eta, params.eigval, alphas, poles);
  const hproots = getroots(params.gamma, eta, params.eigval, alphashp, poleshp);
  root1ptshp = hproots.root1pts; 
  root2ptshp = hproots.root2pts;
//   console.log(root1pts)  
//   console.log(root2pts)


/* subplot 1*/

// Poles trace
const chareqpolesTrace = {
    x: roots.map(r => r[0]),
    y: roots.map(r => r[1]),
    mode: 'markers+text',
    type: 'scatter',
    name: 'Poles',
    marker: { color: jstable? 'green':'red', size: 6, symbol: 'x' },
    text: roots.map((r,i) => `$z_${i+1}$`),
    textposition: 'middle left',  
    textfont: {
        size: 10,        // reduce text size (default is ~12–14)
        color: 'dimgrey'   // optional: set label color
    },
    xaxis : 'x', yaxis: 'y',
};
const nlen = root1pts.re.length;

const root1TraceTraj = {
  x: root1pts.re,
  y: israw ? zerosp : root1pts.im,
  mode: 'lines',
  line: { color: 'yellowgreen', width: 1 },
  opacity: 0.5,
  name: 'root1-trajectory',
  xaxis: 'x', yaxis: 'y'
};

const root1TraceStart = {
  x: [root1pts.re[0]],
  y: israw ? [0] : [root1pts.im[0]],
  mode: 'markers+text',
  marker: { color: 'orangered', size: 3 },
  xaxis: 'x', yaxis: 'y'
};

const root1TraceEnd = {
  x: [root1pts.re[nlen-1]],
  y: israw ? [0] : [root1pts.im[nlen-1]],
  mode: 'markers+text',
  marker: { color: 'orangered', size: 3 },
  xaxis: 'x', yaxis: 'y'
};

const root2TraceTraj = {
  x: root2pts.re,
  y: israw ? zerosp: root2pts.im,
  mode: 'lines',
  line: { color: 'yellowgreen', width: 1 },
  opacity: 0.5,
  name: 'root2-trajectory',
  xaxis: 'x', yaxis: 'y'
};

const root2TraceStart = {
  x: [root2pts.re[0]],
  y: israw ? [0] : [root2pts.im[0]],
  mode: 'markers+text',
  marker: { color: 'blue', size: 3 },
  xaxis: 'x', yaxis: 'y'
};

const root2TraceEnd = {
  x: [root2pts.re[nlen-1]],
  y: israw ? [0] : [root2pts.im[nlen-1]],
  mode: 'markers+text',
  marker: { color: 'blue', size: 3 },
  xaxis: 'x', yaxis: 'y'
};

const root1TraceTrajhp = {
    x: root1ptshp.re,
    y: israw ? zerosph : root1ptshp.im,
    mode: 'lines',
    line: { color: 'crimson', width: 1 },
    opacity: 1,
    name: 'root1-trajectoryhp',
    xaxis: 'x', yaxis: 'y'
  };

const root2TraceTrajhp = {
    x: root2ptshp.re,
    y: israw ? zerosph : root2ptshp.im,
    mode: 'lines',
    line: { color: 'crimson', width: 1 },
    opacity: 1,
    name: 'root2-trajectoryhp',
    xaxis: 'x', yaxis: 'y'
};

// Unit circle for reference
const charequnitCircle = {
x: Array.from({length:361}, (_,k)=>Math.cos(k*Math.PI/180)),
y: Array.from({length:361}, (_,k)=>Math.sin(k*Math.PI/180)),
mode: 'lines',
line: { dash: 'dot', color: 'gray', width:0.25 },
name: 'Unit Circle',
xaxis : 'x', yaxis: 'y',
};

// // Pre-render LaTeX with dynamic values
// const container = document.getElementById("pltheaders");
// container.innerHTML = `$$z^2 - ${a1.toFixed(3)}z + ${a0.toFixed(3)}$$`;
// // MathJax.typesetPromise([container]); // safer async version
const a0fx = israw ? 0 : 3;
const layouta = {
  title: { 
    text: '$\\hbox{overall system dynamics}$',
    font: { size: 12, family: 'Arial, sans-serif', color: 'black' } 
  },
  annotations: [
    {
      text: `$z^2 - ${a1.toFixed(3)}z + ${a0.toFixed(a0fx)}$`,
      xref: "paper", yref: "paper",
      x: 0.5, y: 1.15,
      showarrow: false
  },],
  xaxis: { title: { text: '$\\Re(z)$' }, zeroline: true, range: [-1,1] },
  yaxis: { title: { text: '$\\Im(z)$' }, zeroline: true, range: [-1.2,1.2], scaleanchor: 'x' },
  showlegend: false,
  autosize: true,
};

/* subplot 2*/
const traceCircle = {
    x:unitX, y:unitY, mode:'lines', 
    line:{color:'lightgray', width:0.5},
    xaxis : 'x', yaxis: 'y',
};

const moving = {
    x:[pole], y:[0], mode:'markers+text', 
    marker: {symbol:'x',size:5, color: stable && !highpass? 'green':'red'}, 
    text:['$\\beta_p$'], textposition:'top center',
    textfont: {size:8,},
    xaxis : 'x', yaxis: 'y',
};

const traceTraj = {
    x:poles, y:z, 
    mode:'lines', 
    line:{color:'yellowgreen',width:1}, 
    name:'trajectory',
    xaxis : 'x', yaxis: 'y',
};

const traceTrajhp = {
    x:poleshp, y:z.slice(-nlenph), 
    mode:'lines', 
    line:{color:'crimson',width:1}, 
    name:'trajectoryhp',
    xaxis : 'x', yaxis: 'y',
};


const traceStart = {
    x:[poles[0]], y:[0], mode:'markers+text', 
    marker:{color:'yellowgreen',size:3},
    text:[''], textposition:'top center',
    xaxis : 'x', yaxis: 'y',
};

const traceEnd = {
    x:[poles[nlenp-1]], y:[0], 
    mode:'markers+text', 
    marker:{color:'crimson',size:3},
    text:[''], textposition:'bottom center',
    xaxis : 'x', yaxis: 'y',
};

const layoutb = {
    title: { text: '$\\hbox{change-level\'s single-pole dynamics}$', 
    font: { size: 12, family: 'Arial, sans-serif', color: 'black' } 
  },    
  xaxis:{
    range:[-2,2], 
    title: { 
      text: 'real axis',     
      font: { size: 12,  color: 'darkgray'}, 
    }, 
  }, 
  yaxis:{
    range:[-1,1], 
    title: { 
      text: 'imaginary axis', 
      font: { size: 12, color: 'darkgray' }, 
    }, 
    scaleanchor: 'x'
  },
  annotations: [
      { text: "", xref: "paper", yref: "paper", x: 0.5, y: 1.1, showarrow: false }
  ],
  // width:520, height:250, 
  showlegend:false,
  autosize: true
};

Plotly.newPlot('rightPlotctop',[ charequnitCircle, root1TraceTraj, root1TraceTrajhp, root1TraceStart, root1TraceEnd, root2TraceTraj, root2TraceTrajhp, root2TraceStart, root2TraceEnd, chareqpolesTrace, ], layouta, {displayModeBar:false, responsive: true });

Plotly.newPlot('rightPlotcbottom',[traceCircle, traceTraj, traceTrajhp, moving, traceStart, traceEnd], layoutb, {displayModeBar:false, responsive:true});



// Plotly.relayout('rightPlotctop', {
//   'annotations[0].text': `$z^2 - ${a1.toFixed(3)}z + ${a0.toFixed(3)}$`,
//   'annotations[0].x': 0.5,
//   'annotations[0].y': 1.15,
//   'annotations[0].xref': 'paper',
//   'annotations[0].yref': 'paper'
// });





}


/* Wire controls */
function updateParamsFromInputs(){
  params.input = inputEl.value;

  // if raw
  if(israw){
    betaEl.value = 0;
    gammaEl.value = 0;
  } 
  console.log(israw, betaEl.value, params.beta)
  params.beta = parseFloat(betaEl.value);
  params.gamma = parseFloat(gammaEl.value);
  params.eigval = parseFloat(lamEl.value);
  

  // recompute α bounds and clamp alpha slider range/position
  let {aMin, aMax, eta, aMaxlp, aMaxhpSafe} = lrstablerng(params.beta, params.gamma, params.eigval, true);

  // set alpha slider attributes (some browsers ignore min/max update until reload; we set value/clamp)
  alphaEl.min = aMin; alphaEl.max = aMaxhpSafe;

  /* stable + fast */
  alphaEl.value = 0.99*aMaxlp;
  params.alpha = parseFloat(alphaEl.value);
  
  if (!israw) {   
    /* set gamma, using the other parameters */
    const eal = eta*params.alpha*params.eigval;
    gammaEl.min = -(1-params.beta)/(eal);
    gammaEl.max = params.beta;
    // console.log('eta', eta, 'eal', eal, 'gamma_min', gammaEl.min)

    if(optBtn.disabled){
      const gamma_mopt = -0.9*(1-params.beta)/(eal);
      gammaEl.value = gamma_mopt;
      params.gamma = gamma_mopt;
      // recompute
      ({aMin,aMax, eta, aMaxlp, aMaxhpSafe} = lrstablerng(params.beta, params.gamma, params.eigval, true));
      alphaEl.min = aMin; alphaEl.max = aMaxhpSafe;
      alphaEl.value = 0.99*aMaxlp;
      params.alpha = parseFloat(alphaEl.value);
    }
    // console.log('eta', eta, 'gamma_min', gammaEl.min)
  }

  drawRight();
  drawSys();
}

/* Input listeners */
[lamEl,].forEach(el => {
  el.addEventListener('input', ()=>{
    updateParamsFromInputs();
  });
});
alphaEl.addEventListener('input', ()=>{
  params.alpha = parseFloat(alphaEl.value);
  alphaVal.textContent = params.alpha.toFixed(3);
  drawRight();  
  drawSys();
});
inputEl.addEventListener('change', ()=>{
  params.input = inputEl.value;
  drawSys();
});

// γ slider only works when unlocked
gammaEl.addEventListener('input', () => {
    if (gammafree) {
      params.gamma = parseFloat(gammaEl.value);
      gammaVal.textContent = params.gamma.toFixed(2);
      updateParamsFromInputs();
    }
});

function setgammalisten(beta, gamma) {
    if (phbBtn.disabled) {
        gamma = hbfcn(beta);
    }
    if (nagBtn.disabled) {
        gamma = nagfcn(beta);
    }
    if (sfunBtn.disabled) {
        gamma = sfunfcn(beta);
    }
    return gamma;
}

betaEl.addEventListener('input', () => {
    params.beta = parseFloat(betaEl.value);
   
    params.gamma = setgammalisten(params.beta, params.gamma);

    betaVal.textContent = params.beta.toFixed(2);
    gammaEl.value = params.gamma;
    gammaVal.textContent = params.gamma.toFixed(2);  

    updateParamsFromInputs(); 

});
  

/* Play / Pause / Reset */
let animTimer = null;
function startAnim(){
  if(animTimer) return;
  const {aMin,aMax, eta, aMaxlp, aMaxhpSafe} = lrstablerng(params.beta, params.gamma, params.eigval, true);
  const frames = 360;
  let i = 0;
  animTimer = setInterval(()=>{
    const a = aMin + (aMaxhpSafe - aMin) * (i % frames) / (frames - 1);
    alphaEl.value = a;
    params.alpha = a;
    alphaVal.textContent = params.alpha.toFixed(3);
    drawRight();
    drawSys();
    i++;
  }, 30);
  playBtn.disabled = true;
  pauseBtn.disabled = false;
}
function stopAnim(){
  if(animTimer){ clearInterval(animTimer); animTimer = null; }
  playBtn.disabled = false;
  pauseBtn.disabled = true;
}
playBtn.addEventListener('click', ()=> startAnim());
pauseBtn.addEventListener('click', ()=> stopAnim());
resetBtn.addEventListener('click', ()=>{
  stopAnim();
  betaEl.value = 0.9; 
  gammaEl.value = setgammalisten(parseFloat(betaEl.value), parseFloat(gammaEl.value));
  lamEl.value = 1.0; 
  inputEl.value = 'step';
  updateParamsFromInputs();
});

phbBtn.addEventListener('click', ()=>{
    gammafree = false;
    israw = false;
    freeBtn.disabled = false;
    rawBtn.disabled = false;
    phbBtn.disabled = true;
    nagBtn.disabled = false;
    sfunBtn.disabled = false;
    optBtn.disabled = false;
    gammaEl.value = hbfcn(parseFloat(betaEl.value));
    updateParamsFromInputs();

  });

nagBtn.addEventListener('click', ()=>{
    gammafree = false;
    israw = false;
    freeBtn.disabled = false;
    rawBtn.disabled = false;    
    phbBtn.disabled = false;
    nagBtn.disabled = true;
    sfunBtn.disabled = false;
    optBtn.disabled = false;
    gammaEl.value = nagfcn(parseFloat(betaEl.value));
    updateParamsFromInputs();    
  });
sfunBtn.addEventListener('click', ()=>{
    gammafree = false;
    israw = false;
    freeBtn.disabled = false;
    rawBtn.disabled = false;
    phbBtn.disabled = false;
    nagBtn.disabled = false;
    sfunBtn.disabled = true;    
    optBtn.disabled = false;
    gammaEl.value = sfunfcn(parseFloat(betaEl.value));
    updateParamsFromInputs();
  });
optBtn.addEventListener('click', ()=>{
    gammafree = false;
    israw = false;
    freeBtn.disabled = false;
    rawBtn.disabled = false;
    phbBtn.disabled = false;
    nagBtn.disabled = false;
    sfunBtn.disabled = false;    
    optBtn.disabled = true;
    updateParamsFromInputs();
  });

freeBtn.addEventListener('click', ()=>{
    gammafree = true;
    israw = false;
    freeBtn.disabled = true;
    rawBtn.disabled = false;
    phbBtn.disabled = false;
    nagBtn.disabled = false;
    sfunBtn.disabled = false;
    optBtn.disabled = false;
  });

rawBtn.addEventListener('click', ()=>{
    gammafree = false;
    israw = true;
    freeBtn.disabled = false;
    rawBtn.disabled = true;
    phbBtn.disabled = false;
    nagBtn.disabled = false;
    sfunBtn.disabled = false;
    optBtn.disabled = false;
    updateParamsFromInputs();
  });

/* Init */
betaVal.textContent = params.beta.toFixed(2);
gammaVal.textContent = params.gamma.toFixed(2);
lamVal.textContent = params.eigval.toFixed(2);
alphaVal.textContent = params.alpha.toFixed(3);

drawRight();
drawSys();




// window.addEventListener('DOMContentLoaded', () => {
//     mathbox = MathBox.mathBox({
//     plugins: ["core", "controls", "cursor"],
//     controls: {
//         klass: THREE.OrbitControls,
//     },
//     loop: {
//         start: window == window.top,
//     },
//     camera: {
//         near: 0.01,
//         far: 1000,
//     },
//     });

//     var camera = mathbox.camera({
//         proxy: true,
//         position: [0, 0, 3],
//       });
    
//     var view = mathbox.cartesian({
//         range: [[-2, 2], [-1, 1]],
//         scale: [2, 1],
//       });

//     view
//     .axis({
//         axis: 1,
//         width: 3,
//     })
//     .axis({
//         axis: 2,
//         width: 3,
//     })
//     .grid({
//         width: 2,  
//         divideX: 20,
//         divideY: 10,        
//     });

//     mathbox.select('axis').set('color', 'black');
//     mathbox.set('focus', 3);

//     var data =
//     view.interval({
//       expr: function (emit, x, i, t) {
//         emit(x, Math.sin(x + t));
//       },
//       width: 64,
//       channels: 2,
//     });

//     var curve =
//     view.line({
//       width: 5,
//       color: '#3090FF',
//     });

//     var points =
//     view.point({
//       size: 8,
//       color: '#3090FF',
//     });

//     var vector =
//     view.interval({
//       expr: function (emit, x, i, t) {
//         emit(x, 0);
//         emit(x, -Math.sin(x + t));
//       },
//       width: 64,
//       channels: 2,
//       items: 2,
//     })
//     .vector({
//       end: true,
//       width: 5,
//       color: '#50A000',
//     });

//     var scale =
//     view.scale({
//       divide: 10,
//     });

//     var ticks =
//     view.ticks({
//       width: 5,
//       size: 15,
//       color: 'black',
//     });

//     var format =
//     view.format({
//       digits: 2,
//       weight: 'bold',
//     });

//     var labels =
//     view.label({
//       color: 'red',
//       zIndex: 1,
//     });

//     var play = mathbox.play({
//         target: 'cartesian',
//         pace: 5,
//         to: 2,
//         loop: true,
//         script: [
//           {props: {range: [[-2, 2], [-1, 1]]}},
//           {props: {range: [[-4, 4], [-2, 2]]}},
//           {props: {range: [[-2, 2], [-1, 1]]}},
//         ]
//       });

// });
  