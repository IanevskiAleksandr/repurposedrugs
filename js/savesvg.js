var $jscomp=$jscomp||{};$jscomp.scope={};$jscomp.createTemplateTagFirstArg=function(l){return l.raw=l};$jscomp.createTemplateTagFirstArgWithRaw=function(l,w){l.raw=w;return l};
(function(){var l="undefined"!=typeof exports&&exports||"undefined"!=typeof define&&{}||this||window;"undefined"!==typeof define&&define("save-svg-as-png",[],function(){return l});l["default"]=l;var w=/url\(["']?(.+?)["']?\)/,A={woff2:"font/woff2",woff:"font/woff",otf:"application/x-font-opentype",ttf:"application/x-font-ttf",eot:"application/vnd.ms-fontobject",sfnt:"application/font-sfnt",svg:"image/svg+xml"},y=function(a){if(!(a instanceof HTMLElement||a instanceof SVGElement))throw Error("an HTMLElement or SVGElement is required; got "+
a);},B=function(a){return new Promise(function(c,b){a instanceof HTMLElement||a instanceof SVGElement?c(a):b(Error("an HTMLElement or SVGElement is required; got "+a))})},F=function(a){return a&&0===a.lastIndexOf("http",0)&&-1===a.lastIndexOf(window.location.host)},G=function(a){var c=Object.keys(A).filter(function(b){return 0<a.indexOf("."+b)}).map(function(b){return A[b]});if(c)return c[0];console.error("Unknown font format for "+a+". Fonts may not be working correctly.");return"application/octet-stream"},
C=function(a,c,b){a=a.viewBox&&a.viewBox.baseVal&&a.viewBox.baseVal[b]||null!==c.getAttribute(b)&&!c.getAttribute(b).match(/%$/)&&parseInt(c.getAttribute(b))||a.getBoundingClientRect()[b]||parseInt(c.style[b])||parseInt(window.getComputedStyle(a).getPropertyValue(b));return"undefined"===typeof a||null===a||isNaN(parseFloat(a))?0:a},H=function(a,c,b,d){if("svg"===a.tagName)return{width:b||C(a,c,"width"),height:d||C(a,c,"height")};if(a.getBBox)return a=a.getBBox(),{width:a.x+a.width,height:a.y+a.height}},
I=function(a){return decodeURIComponent(encodeURIComponent(a).replace(/%([0-9A-F]{2})/g,function(c,b){var d=String.fromCharCode("0x"+b);return"%"===d?"%25":d}))},D=function(a){var c=window.atob(a.split(",")[1]);a=a.split(",")[0].split(":")[1].split(";")[0];for(var b=new ArrayBuffer(c.length),d=new Uint8Array(b),f=0;f<c.length;f++)d[f]=c.charCodeAt(f);return new Blob([b],{type:a})},J=function(a){return Promise.all(Array.from(a.querySelectorAll("image")).map(function(c){var b=c.getAttributeNS("http://www.w3.org/1999/xlink",
"href")||c.getAttribute("href");if(!b)return Promise.resolve(null);F(b)&&(b+=(-1===b.indexOf("?")?"?":"&")+"t="+(new Date).valueOf());return new Promise(function(d,f){var g=document.createElement("canvas"),m=new Image;m.crossOrigin="anonymous";m.src=b;m.onerror=function(){return f(Error("Could not load "+b))};m.onload=function(){g.width=m.width;g.height=m.height;g.getContext("2d").drawImage(m,0,0);c.setAttributeNS("http://www.w3.org/1999/xlink","href",g.toDataURL("image/png"));d(!0)}})}))},x={},K=
function(a){return Promise.all(a.map(function(c){return new Promise(function(b,d){if(x[c.url])return b(x[c.url]);var f=new XMLHttpRequest;f.addEventListener("load",function(){var g="";for(var m=new Uint8Array(f.response),t=0;t<m.byteLength;t++)g+=String.fromCharCode(m[t]);g=window.btoa(g);g=c.text.replace(w,'url("data:'+c.format+";base64,"+g+'")')+"\n";x[c.url]=g;b(g)});f.addEventListener("error",function(g){console.warn("Failed to load font from: "+c.url,g);x[c.url]=null;b(null)});f.addEventListener("abort",
function(g){console.warn("Aborted loading font from: "+c.url,g);b(null)});f.open("GET",c.url);f.responseType="arraybuffer";f.send()})})).then(function(c){return c.filter(function(b){return b}).join("")})},z=null,L=function(){return z?z:z=Array.from(document.styleSheets).map(function(a){try{return{rules:a.cssRules,href:a.href}}catch(c){return console.warn("Stylesheet could not be loaded: "+a.href,c),{}}})},N=function(a,c){var b=c||{},d=b.selectorRemap,f=b.modifyStyle,g=b.fonts,m=b.excludeUnusedCss,
t=b.modifyCss||function(e,n){var p=d?d(e):e,k=f?f(n):n;return p+"{"+k+"}\n"},q=[],u="undefined"===typeof g,v=g||[];L().forEach(function(e){var n=e.rules,p=e.href;n&&Array.from(n).forEach(function(k){if("undefined"!=typeof k.style){a:{var r=k.selectorText;if(r)try{var h=a.querySelector(r)||a.parentNode&&a.parentNode.querySelector(r);break a}catch(M){console.warn('Invalid CSS selector "'+r+'"',M)}h=void 0}h?q.push(t(k.selectorText,k.style.cssText)):u&&k.cssText.match(/^@font-face/)?(h=(h=k.cssText.match(w))&&
h[1]||"",!h||h.match(/^data:/)||"about:blank"===h?k=void 0:(h=h.startsWith("../")?p+"/../"+h:h.startsWith("./")?p+"/."+h:h,k={text:k.cssText,format:G(h),url:h}),k&&v.push(k)):m||q.push(k.cssText)}})});return K(v).then(function(e){return q.join("\n")+e})},E=function(){if(!(navigator.msSaveOrOpenBlob||"download"in document.createElement("a")))return{popup:window.open()}};l.prepareSvg=function(a,c,b){y(a);var d=c||{},f=void 0===d.left?0:d.left,g=void 0===d.top?0:d.top,m=d.width,t=d.height,q=void 0===
d.scale?1:d.scale,u=void 0===d.responsive?!1:d.responsive,v=void 0===d.excludeCss?!1:d.excludeCss;return J(a).then(function(){var e=a.cloneNode(!0);e.style.backgroundColor=(c||{}).backgroundColor||a.style.backgroundColor;var n=H(a,e,m,t),p=n.width,k=n.height;if("svg"!==a.tagName)if(a.getBBox)null!=e.getAttribute("transform")&&e.setAttribute("transform",e.getAttribute("transform").replace(/translate\(.*?\)/,"")),n=document.createElementNS("http://www.w3.org/2000/svg","svg"),n.appendChild(e),e=n;else{console.error("Attempted to render non-SVG element",
a);return}e.setAttribute("version","1.1");e.setAttribute("viewBox",[f,g,p,k].join(" "));e.getAttribute("xmlns")||e.setAttributeNS("http://www.w3.org/2000/xmlns/","xmlns","http://www.w3.org/2000/svg");e.getAttribute("xmlns:xlink")||e.setAttributeNS("http://www.w3.org/2000/xmlns/","xmlns:xlink","http://www.w3.org/1999/xlink");u?(e.removeAttribute("width"),e.removeAttribute("height"),e.setAttribute("preserveAspectRatio","xMinYMin meet")):(e.setAttribute("width",p*q),e.setAttribute("height",k*q));Array.from(e.querySelectorAll("foreignObject > *")).forEach(function(r){r.setAttributeNS("http://www.w3.org/2000/xmlns/",
"xmlns","svg"===r.tagName?"http://www.w3.org/2000/svg":"http://www.w3.org/1999/xhtml")});if(v)if(n=document.createElement("div"),n.appendChild(e),n=n.innerHTML,"function"===typeof b)b(n,p,k);else return{src:n,width:p,height:k};else return N(a,c).then(function(r){var h=document.createElement("style");h.setAttribute("type","text/css");h.innerHTML="<![CDATA[\n"+r+"\n]]\x3e";r=document.createElement("defs");r.appendChild(h);e.insertBefore(r,e.firstChild);h=document.createElement("div");h.appendChild(e);
h=h.innerHTML.replace(/NS\d+:href/gi,'xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href');if("function"===typeof b)b(h,p,k);else return{src:h,width:p,height:k}})})};l.svgAsDataUri=function(a,c,b){y(a);return l.prepareSvg(a,c).then(function(d){var f=d.width,g=d.height;d="data:image/svg+xml;base64,"+window.btoa(I('<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [<!ENTITY nbsp "&#160;">]>'+d.src));"function"===
typeof b&&b(d,f,g);return d})};l.svgAsPngUri=function(a,c,b){y(a);var d=c||{},f=void 0===d.encoderType?"image/png":d.encoderType,g=void 0===d.encoderOptions?.8:d.encoderOptions,m=d.canvg,t=function(q){var u=q.src,v=q.width;q=q.height;var e=document.createElement("canvas"),n=e.getContext("2d"),p=window.devicePixelRatio||1;e.width=v*p;e.height=q*p;e.style.width=e.width+"px";e.style.height=e.height+"px";n.setTransform(p,0,0,p,0,0);m?m(e,u):n.drawImage(u,0,0);try{var k=e.toDataURL(f,g)}catch(r){if("undefined"!==
typeof SecurityError&&r instanceof SecurityError||"SecurityError"===r.name){console.error("Rendered SVG images cannot be downloaded in this browser.");return}throw r;}"function"===typeof b&&b(k,e.width,e.height);return Promise.resolve(k)};return m?l.prepareSvg(a,c).then(t):l.svgAsDataUri(a,c).then(function(q){return new Promise(function(u,v){var e=new Image;e.onload=function(){return u(t({src:e,width:e.width,height:e.height}))};e.onerror=function(){v("There was an error loading the data URI as an image on the following SVG\n"+
window.atob(q.slice(26))+"Open the following link to see browser's diagnosis\n"+q)};e.src=q})})};l.download=function(a,c,b){if(navigator.msSaveOrOpenBlob)navigator.msSaveOrOpenBlob(D(c),a);else{var d=document.createElement("a");if("download"in d){d.download=a;d.style.display="none";document.body.appendChild(d);try{var f=D(c),g=URL.createObjectURL(f);d.href=g;d.onclick=function(){return requestAnimationFrame(function(){return URL.revokeObjectURL(g)})}}catch(m){console.error(m),console.warn("Error while getting object URL. Falling back to string URL."),
d.href=c}d.click();document.body.removeChild(d)}else b&&b.popup&&(b.popup.document.title=a,b.popup.location.replace(c))}};l.saveSvg=function(a,c,b){var d=E();return B(a).then(function(f){return l.svgAsDataUri(f,b||{})}).then(function(f){return l.download(c,f,d)})};l.saveSvgAsPng=function(a,c,b){var d=E();return B(a).then(function(f){return l.svgAsPngUri(f,b||{})}).then(function(f){return l.download(c,f,d)})}})();