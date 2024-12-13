let humanScore = 0;
let computerScore = 0;
let currentRoundNumber = 1;

// Write your code below:
const generateTarget = () => {
    return Math.floor(Math.random() * 10);
}

const compareGuesses = (userGuess, compGuess, targetNum) => {
    const humanDiff = Math.abs(userGuess - targetNum);
    const compDiff = Math.abs(compGuess - targetNum);

    if (humanDiff < compDiff) {
        return true;
    } else if (humanDiff > compDiff) {
        return false;
    } else {
        return true;
    }
}

const updateScore = (winner) => {
    if (winner === 'human') {
        humanScore++;
    } else if (winner === 'computer') {
        computerScore++;
    }
}

const advanceRound = () => {
    currentRoundNumber++;
}