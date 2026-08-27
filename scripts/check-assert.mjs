export function createCheck(name) {
  const errors = [];

  return {
    expect(condition, label) {
      if (!condition) errors.push(label);
    },
    expectEqual(actual, expected, label) {
      if (actual !== expected) {
        errors.push(`${label} Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.`);
      }
    },
    expectDeepEqual(actual, expected, label) {
      const actualText = JSON.stringify(actual);
      const expectedText = JSON.stringify(expected);
      if (actualText !== expectedText) {
        errors.push(`${label} Expected ${expectedText}, got ${actualText}.`);
      }
    },
    expectTypedArray(value, constructor, label) {
      if (!(value instanceof constructor)) {
        errors.push(`${label} Expected ${constructor.name}, got ${value?.constructor?.name || typeof value}.`);
      }
    },
    expectThrows(fn, messagePart, label) {
      try {
        fn();
        errors.push(`${label} Expected an error containing ${JSON.stringify(messagePart)}.`);
      } catch (error) {
        if (!String(error?.message || error).includes(messagePart)) {
          errors.push(`${label} Expected an error containing ${JSON.stringify(messagePart)}, got ${JSON.stringify(error?.message || String(error))}.`);
        }
      }
    },
    done(successMessage) {
      if (errors.length) {
        console.error(`${name} failed:`);
        for (const error of errors) {
          console.error(`- ${error}`);
        }
        process.exit(1);
      }
      console.log(successMessage || `${name} passed.`);
    },
  };
}
