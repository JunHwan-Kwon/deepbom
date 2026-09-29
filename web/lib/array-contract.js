// Element identity only. Numeric evidence with NaN/signed zero uses
// numericArraysExactlyEqual (Object.is) instead of this shape/reference rule.
export function arraysEqual(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (!Object.hasOwn(left, index) || !Object.hasOwn(right, index) || left[index] !== right[index]) return false;
  }
  return true;
}
