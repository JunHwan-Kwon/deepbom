import { deflateSync } from "node:zlib";

import { buildModelIrVisualizationBundle } from "../web/lib/model-ir-visualization.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";

const FONT = Object.freeze({
  " ":[0,0,0,0,0,0,0],"#":[10,31,10,10,31,10,0],"-":[0,0,0,31,0,0,0],".":[0,0,0,0,0,12,12],"/":[1,2,4,8,16,0,0],":":[0,12,12,0,12,12,0],"_":[0,0,0,0,0,0,31],"?":[14,17,1,2,4,0,4],"|":[0,4,4,4,4,4,0],
  "0":[14,17,19,21,25,17,14],"1":[4,12,4,4,4,4,14],"2":[14,17,1,2,4,8,31],"3":[30,1,1,14,1,1,30],"4":[2,6,10,18,31,2,2],"5":[31,16,16,30,1,1,30],"6":[14,16,16,30,17,17,14],"7":[31,1,2,4,8,8,8],"8":[14,17,17,14,17,17,14],"9":[14,17,17,15,1,1,14],
  A:[14,17,17,31,17,17,17],B:[30,17,17,30,17,17,30],C:[14,17,16,16,16,17,14],D:[30,17,17,17,17,17,30],E:[31,16,16,30,16,16,31],F:[31,16,16,30,16,16,16],G:[14,17,16,23,17,17,15],H:[17,17,17,31,17,17,17],I:[14,4,4,4,4,4,14],J:[7,2,2,2,2,18,12],K:[17,18,20,24,20,18,17],L:[16,16,16,16,16,16,31],M:[17,27,21,21,17,17,17],N:[17,25,21,19,17,17,17],O:[14,17,17,17,17,17,14],P:[30,17,17,30,16,16,16],Q:[14,17,17,17,21,18,13],R:[30,17,17,30,20,18,17],S:[15,16,16,14,1,1,30],T:[31,4,4,4,4,4,4],U:[17,17,17,17,17,17,14],V:[17,17,17,17,17,10,4],W:[17,17,17,21,21,21,10],X:[17,17,10,4,10,17,17],Y:[17,17,10,4,4,4,4],Z:[31,1,2,4,8,16,31],
});

export function exportModelIrVisualizationPngFiles(modelIr, options = {}) {
  const bundle = buildModelIrVisualizationBundle(modelIr, options);
  return bundle.pages.map((page) => {
    const bytes = renderPage(page, modelIr, options.orientation || bundle.manifest.orientation);
    return {
      name: page.filename.replace(/\.svg$/, ".png"),
      data: bytes,
      media_type: "image/png",
      sha256: sha256BytesHex(bytes),
      source_svg_sha256: page.sha256,
      derivation: "deepbom_builtin_5x7_font_black_white_png_300dpi_v1",
    };
  });
}

function renderPage(page, modelIr, orientation) {
  const width = orientation === "landscape" ? 3508 : 2480;
  const height = orientation === "landscape" ? 2480 : 3508;
  const pixels = new Uint8Array(width * height * 4);
  fill(pixels, [255, 255, 255, 255]);
  const margin = 90;
  rectangle(pixels, width, height, 55, 55, width - 110, height - 110, false);
  drawText(pixels, width, height, margin, 100, `${page.render_model.projection.level} ${page.render_model.projection.title}`, 4);
  drawText(pixels, width, height, margin, 145, modelIr.artifact.filename, 3);
  drawText(pixels, width, height, margin, 180, `ARTIFACT SHA256 ${modelIr.artifact.sha256.slice(0, 16)} PAGE ${page.page_number} OF ${page.page_count}`, 2);
  const rows = page.render_model.rows;
  const rowHeight = Math.floor((height - 390) / Math.max(1, orientation === "landscape" ? 12 : 22));
  for (const [index, row] of rows.entries()) {
    const y = 230 + index * rowHeight;
    rectangle(pixels, width, height, margin, y, width - margin * 2, rowHeight - 8, dashed(row.evidence_class));
    drawText(pixels, width, height, margin + 14, y + 10, `${row.kind} | ${row.title}`.slice(0, orientation === "landscape" ? 112 : 76), 5);
    drawText(pixels, width, height, margin + 14, y + 52, row.detail.slice(0, orientation === "landscape" ? 112 : 76), 5);
    drawText(pixels, width, height, margin + 14, y + 94, row.subject_ref.slice(0, orientation === "landscape" ? 112 : 76), 5);
  }
  drawText(pixels, width, height, margin, height - 105, `DEEPBOM MODEL IR ${modelIr.method_version} ${modelIr.model_ir_sha256.slice(0, 16)}`, 2);
  return encodePng(width, height, pixels);
}

function dashed(value) { return /DERIVED|PREDICTED|NOT_ASSESSABLE|INFERRED/i.test(String(value || "")); }
function fill(pixels, color) { for (let offset = 0; offset < pixels.length; offset += 4) pixels.set(color, offset); }
function setPixel(pixels, width, height, x, y, black = true) { if (x < 0 || y < 0 || x >= width || y >= height) return; pixels.set(black ? [0,0,0,255] : [255,255,255,255], (y * width + x) * 4); }
function line(pixels,width,height,x0,y0,x1,y1,dash=false){let dx=Math.abs(x1-x0),sx=x0<x1?1:-1,dy=-Math.abs(y1-y0),sy=y0<y1?1:-1,error=dx+dy,step=0;while(true){if(!dash||Math.floor(step/8)%2===0)setPixel(pixels,width,height,x0,y0);if(x0===x1&&y0===y1)break;const twice=2*error;if(twice>=dy){error+=dy;x0+=sx}if(twice<=dx){error+=dx;y0+=sy}step+=1}}
function rectangle(pixels,width,height,x,y,w,h,dash){line(pixels,width,height,x,y,x+w-1,y,dash);line(pixels,width,height,x,y+h-1,x+w-1,y+h-1,dash);line(pixels,width,height,x,y,x,y+h-1,dash);line(pixels,width,height,x+w-1,y,x+w-1,y+h-1,dash)}
function drawText(pixels,width,height,x,y,value,scale){let cursor=x;for(const raw of String(value).toUpperCase()){const glyph=FONT[raw]||FONT["?"];for(let row=0;row<7;row+=1)for(let column=0;column<5;column+=1)if(glyph[row]&(1<<(4-column)))for(let yy=0;yy<scale;yy+=1)for(let xx=0;xx<scale;xx+=1)setPixel(pixels,width,height,cursor+column*scale+xx,y+row*scale+yy);cursor+=6*scale;if(cursor>=width-80)break}}
function encodePng(width,height,pixels){const stride=width*4,raw=Buffer.alloc((stride+1)*height);for(let row=0;row<height;row+=1){const target=row*(stride+1);raw[target]=0;Buffer.from(pixels.buffer,pixels.byteOffset+row*stride,stride).copy(raw,target+1)}const signature=Buffer.from([137,80,78,71,13,10,26,10]),ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width,0);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=6;return Buffer.concat([signature,chunk("IHDR",ihdr),chunk("IDAT",deflateSync(raw,{level:9})),chunk("IEND",Buffer.alloc(0))])}
function chunk(type,data){const name=Buffer.from(type,"ascii"),body=Buffer.concat([name,data]),result=Buffer.alloc(12+data.length);result.writeUInt32BE(data.length,0);body.copy(result,4);result.writeUInt32BE(crc32(body),8+data.length);return result}
function crc32(bytes){let crc=0xffffffff;for(const byte of bytes){crc^=byte;for(let bit=0;bit<8;bit+=1)crc=(crc>>>1)^((crc&1)?0xedb88320:0)}return(crc^0xffffffff)>>>0}
