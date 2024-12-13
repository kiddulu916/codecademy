let raceNumber = Math.floor(Math.random() * 1000);
let registeredEarly = true;
let runnerAge = 18;

if (registeredEarly && runnerAge > 18) {
  raceNumber += 1000;
}

if (registeredEarly && runnerAge > 18) {
  console.log(`Your race number is ${raceNumber}. You will race at 9:30 am.`);
} else if (!registeredEarly && runnerAge > 18) {
  console.log(`Your race number is ${raceNumber}. You will race at 11:00 am.`);
} else if (runnerAge < 18) {
  console.log(`Your race number is ${raceNumber}. You will race at 12:30 am.`);
} else {
  console.log('Please see the registration desk.');
}