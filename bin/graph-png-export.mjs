import { deflateSync } from "node:zlib";

import { buildGraphVisualizationScene, VISUALIZATION_MANIFEST_SCHEMA } from "../web/lib/graph-export.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const MAX_DIMENSION = 4096;
const FONT = Object.freeze({
  " ":[0,0,0,0,0,0,0],"#":[10,31,10,10,31,10,0],"-":[0,0,0,31,0,0,0],".":[0,0,0,0,0,12,12],"/":[1,2,4,8,16,0,0],":":[0,12,12,0,12,12,0],"_":[0,0,0,0,0,0,31],
  "0":[14,17,19,21,25,17,14],"1":[4,12,4,4,4,4,14],"2":[14,17,1,2,4,8,31],"3":[30,1,1,14,1,1,30],"4":[2,6,10,18,31,2,2],"5":[31,16,16,30,1,1,30],"6":[14,16,16,30,17,17,14],"7":[31,1,2,4,8,8,8],"8":[14,17,17,14,17,17,14],"9":[14,17,17,15,1,1,14],
  A:[14,17,17,31,17,17,17],B:[30,17,17,30,17,17,30],C:[14,17,16,16,16,17,14],D:[30,17,17,17,17,17,30],E:[31,16,16,30,16,16,31],F:[31,16,16,30,16,16,16],G:[14,17,16,23,17,17,15],H:[17,17,17,31,17,17,17],I:[14,4,4,4,4,4,14],J:[7,2,2,2,2,18,12],K:[17,18,20,24,20,18,17],L:[16,16,16,16,16,16,31],M:[17,27,21,21,17,17,17],N:[17,25,21,19,17,17,17],O:[14,17,17,17,17,17,14],P:[30,17,17,30,16,16,16],Q:[14,17,17,17,21,18,13],R:[30,17,17,30,20,18,17],S:[15,16,16,14,1,1,30],T:[31,4,4,4,4,4,4],U:[17,17,17,17,17,17,14],V:[17,17,17,17,17,10,4],W:[17,17,17,21,21,21,10],X:[17,17,10,4,10,17,17],Y:[17,17,10,4,4,4,4],Z:[31,1,2,4,8,16,31],
});

export function exportGraphPng(graph, { view = "structure" } = {}) {
  const visual = buildGraphVisualizationScene(graph, view);
  const projection = visual.projection;
  const sourceScene = visual.scene;
  const sourceScale = Math.min(1, MAX_DIMENSION / sourceScene.width, MAX_DIMENSION / sourceScene.height);
  const rasterLayout = sourceScale < 0.72 ? tiledRasterLayout(sourceScene) : { scene: sourceScene, kind: "graph_topology" };
  const scene = rasterLayout.scene;
  const scale = Math.min(1, MAX_DIMENSION / scene.width, MAX_DIMENSION / scene.height);
  const width = Math.max(1, Math.round(scene.width * scale));
  const height = Math.max(1, Math.round(scene.height * scale));
  const pixels = new Uint8Array(width * height * 4);
  fill(pixels, width, height, [245, 242, 234, 255]);
  const scaled = (value) => Math.round(value * scale);
  for (const edge of scene.edges) {
    const from = scene.positions.get(edge.from), to = scene.positions.get(edge.to);
    if (!from || !to) continue;
    line(pixels, width, height, scaled(from.x + 124), scaled(from.y + 94), scaled(to.x + 124), scaled(to.y), [141, 136, 125, 255]);
  }
  for (const node of scene.nodes) {
    const placement = String(node.placement?.status || "");
    const risk = view === "quantization" && node.quantization?.risk && node.quantization.risk !== "none";
    const accelerator = placement.includes("DELEGATABLE") || placement.includes("ELIGIBLE") || placement.includes("ACCELERATOR");
    const fallback = placement.includes("FALLBACK") || placement.includes("OTHER_POOL");
    const border = risk || fallback ? [180, 76, 62, 255] : accelerator ? [23, 125, 107, 255] : [119, 112, 100, 255];
    const background = risk ? [255, 241, 237, 255] : accelerator ? [237, 247, 243, 255] : [255, 253, 250, 255];
    const x=scaled(node.x),y=scaled(node.y),w=Math.max(2,scaled(248)),h=Math.max(2,scaled(94));
    rectangle(pixels,width,height,x,y,w,h,background,border);
    const fontScale=scale>=.9?2:1;
    const textLimit=fontScale===2?18:34;
    drawText(pixels,width,height,x+Math.max(2,scaled(10)),y+Math.max(2,scaled(9)),`#${String(node.index).padStart(3,"0")} ${node.domain}`.slice(0,textLimit),fontScale,[91,84,75,255]);
    drawText(pixels,width,height,x+Math.max(2,scaled(10)),y+Math.max(10,scaled(35)),String(node.label).slice(0,textLimit),fontScale,[25,23,20,255]);
    drawText(pixels,width,height,x+Math.max(2,scaled(10)),y+Math.max(18,scaled(61)),placement.slice(0,textLimit),fontScale,border);
  }
  drawText(pixels,width,height,scaled(48),scaled(24),`${graph.artifact.filename} ${view}`.slice(0,80),Math.max(1,Math.floor(scale*2)),[25,23,20,255]);
  const manifestBody={schema:VISUALIZATION_MANIFEST_SCHEMA,graph_ir_sha256:graph.graph_ir_sha256,view,format:"png",full_node_count:graph.nodes.length,rendered_node_count:projection.nodes.length,rendered_edge_count:projection.edges.length,level_of_detail:projection.level,executable_dag_claim:graph.projection.executable_dag_claim,dimensions:{width,height},source_scene_dimensions:{width:sourceScene.width,height:sourceScene.height},raster_layout:rasterLayout.kind,raster_layout_dimensions:{width:scene.width,height:scene.height},raster_scale:Number(scale.toFixed(12)),determinism:"canonical_graph_ir_fixed_layout_builtin_5x7_font_rgba_deflate_v2",interpretation_boundary:graph.interpretation_boundary};
  const manifest={...manifestBody,visualization_manifest_sha256:sha256TextHex(canonicalJson(manifestBody))};
  return { bytes: encodePng(width,height,pixels), mediaType:"image/png", manifest };
}

function tiledRasterLayout(source) {
  const nodeWidth = 248;
  const nodeHeight = 94;
  const gapX = 24;
  const gapY = 20;
  const margin = 32;
  const header = 72;
  const maximumColumns = Math.max(1, Math.floor((MAX_DIMENSION - margin * 2 + gapX) / (nodeWidth + gapX)));
  const columns = Math.min(maximumColumns, Math.max(1, Math.ceil(Math.sqrt(source.nodes.length * 1.6))));
  const positions = new Map();
  const nodes = source.nodes.map((node, index) => {
    const x = margin + (index % columns) * (nodeWidth + gapX);
    const y = margin + header + Math.floor(index / columns) * (nodeHeight + gapY);
    positions.set(node.id, { x, y });
    return { ...node, x, y };
  });
  const rows = Math.ceil(nodes.length / columns);
  return {
    kind: "readable_row_major_poster",
    scene: {
      ...source,
      nodes,
      positions,
      width: margin * 2 + columns * nodeWidth + Math.max(0, columns - 1) * gapX,
      height: margin * 2 + header + rows * nodeHeight + Math.max(0, rows - 1) * gapY,
    },
  };
}

function fill(pixels,width,height,color){for(let offset=0;offset<width*height*4;offset+=4)pixels.set(color,offset)}
function setPixel(pixels,width,height,x,y,color){if(x<0||y<0||x>=width||y>=height)return;pixels.set(color,(y*width+x)*4)}
function line(pixels,width,height,x0,y0,x1,y1,color){let dx=Math.abs(x1-x0),sx=x0<x1?1:-1,dy=-Math.abs(y1-y0),sy=y0<y1?1:-1,error=dx+dy;while(true){setPixel(pixels,width,height,x0,y0,color);if(x0===x1&&y0===y1)break;const twice=2*error;if(twice>=dy){error+=dy;x0+=sx}if(twice<=dx){error+=dx;y0+=sy}}}
function rectangle(pixels,width,height,x,y,w,h,background,border){for(let row=Math.max(0,y);row<Math.min(height,y+h);row+=1)for(let column=Math.max(0,x);column<Math.min(width,x+w);column+=1)setPixel(pixels,width,height,column,row,background);line(pixels,width,height,x,y,x+w-1,y,border);line(pixels,width,height,x,y+h-1,x+w-1,y+h-1,border);line(pixels,width,height,x,y,x,y+h-1,border);line(pixels,width,height,x+w-1,y,x+w-1,y+h-1,border)}
function drawText(pixels,width,height,x,y,value,scale,color){let cursor=x;for(const raw of String(value).toUpperCase()){const glyph=FONT[raw]||FONT[" "];for(let row=0;row<7;row+=1)for(let column=0;column<5;column+=1)if(glyph[row]&(1<<(4-column)))for(let yy=0;yy<scale;yy+=1)for(let xx=0;xx<scale;xx+=1)setPixel(pixels,width,height,cursor+column*scale+xx,y+row*scale+yy,color);cursor+=6*scale;if(cursor>=width-6*scale)break}}
function encodePng(width,height,pixels){const stride=width*4,raw=Buffer.alloc((stride+1)*height);for(let row=0;row<height;row+=1){const target=row*(stride+1);raw[target]=0;Buffer.from(pixels.buffer,pixels.byteOffset+row*stride,stride).copy(raw,target+1)}const signature=Buffer.from([137,80,78,71,13,10,26,10]),ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width,0);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=6;return Buffer.concat([signature,chunk("IHDR",ihdr),chunk("IDAT",deflateSync(raw,{level:9})),chunk("IEND",Buffer.alloc(0))])}
function chunk(type,data){const name=Buffer.from(type,"ascii"),body=Buffer.concat([name,data]),result=Buffer.alloc(12+data.length);result.writeUInt32BE(data.length,0);body.copy(result,4);result.writeUInt32BE(crc32(body),8+data.length);return result}
function crc32(bytes){let crc=0xffffffff;for(const byte of bytes){crc^=byte;for(let bit=0;bit<8;bit+=1)crc=(crc>>>1)^((crc&1)?0xedb88320:0)}return(crc^0xffffffff)>>>0}
