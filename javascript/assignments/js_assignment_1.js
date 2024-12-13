// setting a variable to a number
const kelvin = 293;

// converting kelvin to celsius
const celsius = kelvin - 273;

// converting celsius to fahrenheit
let fahrenheit = celsius * (9/5) + 32;

// rounding fahrenheit
fahrenheit = Math.floor(fahrenheit);

// output of fahrenheit rounded
console.log(`The temperature is ${fahrenheit} degrees Fahrenheit.`);

// converting celsius to Newton
let newton = celsius * (33/100);

// rounding newton
newton = Math.floor(newton);

// output of newton rounded
console.log(`The temperature is ${newton} degrees Newton.`);
