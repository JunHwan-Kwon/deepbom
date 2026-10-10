export const EXACT_RATIO_DECIMAL_DIGITS = 15;

function nonnegativeInteger(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  const text = value == null ? "" : String(value);
  return /^(?:0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
}

export function exactNonnegativeRatio(numeratorValue, denominatorValue, digits = EXACT_RATIO_DECIMAL_DIGITS) {
  const numerator = nonnegativeInteger(numeratorValue);
  const denominator = nonnegativeInteger(denominatorValue);
  if (numerator == null || denominator == null || denominator === 0n
    || !Number.isSafeInteger(digits) || digits < 0 || digits > 15) return null;
  const scale = 10n ** BigInt(digits);
  const scaled = (numerator * scale + denominator / 2n) / denominator;
  const value = Number(scaled) / (10 ** digits);
  return Number.isFinite(value) ? value : null;
}

// Arithmetic on the canonical decimal spelling of a reported finite JSON number.
// This does not recover precision that the producer omitted before serialization.
export function reportedDecimal(value) {
  if(typeof value !== 'number' || !Number.isFinite(value))throw Error('Reported decimal requires a finite number');
  const m=/^(-?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(String(value));
  return {coefficient:BigInt(m[1]+m[2]+(m[3]||'')),exponent:Number(m[4]||0)-(m[3]||'').length};
}
export function addExactDecimals(a,b){
  const exponent=Math.min(a.exponent,b.exponent);
  return {coefficient:a.coefficient*10n**BigInt(a.exponent-exponent)+b.coefficient*10n**BigInt(b.exponent-exponent),exponent};
}
export function negateExactDecimal(value){return {...value,coefficient:-value.coefficient};}
export function subtractExactDecimals(a,b){return addExactDecimals(a,negateExactDecimal(b));}
export function halveExactDecimal(value){return {coefficient:value.coefficient*5n,exponent:value.exponent-1};}
export function compareExactDecimals(a,b){const d=subtractExactDecimals(a,b).coefficient;return d<0n?-1:d>0n?1:0;}
export function exactDecimalText(value){
  if(value.coefficient===0n)return '0';
  const negative=value.coefficient<0n,digits=(negative?-value.coefficient:value.coefficient).toString(),point=digits.length+value.exponent;
  let text=point<=0?'0.'+'0'.repeat(-point)+digits:point>=digits.length?digits+'0'.repeat(point-digits.length):digits.slice(0,point)+'.'+digits.slice(point);
  if(text.includes('.'))text=text.replace(/0+$/,'').replace(/\.$/,'');
  return (negative?'-':'')+text;
}
export function exactDecimalNumber(value){const number=Number(exactDecimalText(value));return Number.isFinite(number)&&(number!==0||value.coefficient===0n)?number:null;}
