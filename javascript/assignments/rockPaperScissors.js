console.log('hi');

const getUsersChoice = (userInput) => {
  userInput.toLowerCase();
  if(userInput === "rock" || userInput === "paper" || userInput === "scissors" || userInput === "bomb") {
    console.log("User Chose:")
    return userInput;
  }
  else {
    console.log("Invalid input. Please choose between 'rock', 'paper', or 'scissors'");
  }
}

function getComputerChoice() {
  var compChoice = Math.floor(Math.random() * 3);
  if(compChoice === 0) {
    console.log("Computer chose:");
    return 'rock';
  }
  if(compChoice === 1) {
    console.log("Computer chose:");
    return 'paper';
  }
  else {
    console.log("Computer chose:");
    return 'scissors';
  }
}

function determineWinner(usersChoice, computersChoice) {
  if(usersChoice === computersChoice) {
    return "Tie";
  }
  if(usersChoice === 'rock') {
    if(computersChoice === 'paper') {
      return "Computer wins! Paper beats Rock!";
    }
    else {
      return "User Wins! Rock smashes Scissors!";
    }
  }
  if(usersChoice === 'paper') {
    if(computersChoice === 'scissors') {
      return "Computer wins! Scissors cut Paper!";
    }
    else {
      return "User Wins! Paper covers Rock!";
    }
  }
  if(usersChoice === 'scissors') {
    if(computersChoice === 'rock') {
      return "Computer wins! Rock crushes Scissors";
    }
    else {
      return "User Wins! Scissors cut Paper!";
    }
  }
  if(usersChoice === 'bomb') {
    return "User Wins! Bomb beats everything!"
  }
}

function playGame() {
  var userChoice = getUsersChoice('paper')
  console.log(userChoice)
  var compChoice = getComputerChoice()
  console.log(compChoice)
  console.log(determineWinner(userChoice, compChoice)) 
}

playGame()