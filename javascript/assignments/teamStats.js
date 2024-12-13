const team = {
    _players: [
        { firstName: 'John', lastName: 'Doe', age: 24 },
        { firstName: 'Joe', lastName: 'Dawn', age: 25 },
        { firstName: 'Corey', lastName: 'Hilsenbeck', age: 26 }
    ],
    
    _games: [
        { opponent: 'Broncos', teamPoints: 42, opponentPoints: 27 },
        { opponent: 'Rams', teamPoints: 35, opponentPoints: 37 },
        { opponent: 'Cowboys', teamPoints: 41, opponentPoints: 40 }
    ],
    
    get players() {
        return this._players;
    },

    get games() {
        return this._games;
    },

    addPlayer(newFirstName, newLastName, newAge) {
        const player = {
            firstName: newFirstName,
            lastName: newLastName,
            age: newAge
        }
        this._players.push(player)
    }

    addGame(opponent, teamPoints, opponentPoints) {
        const game = {
            opponent: opponent,
            teamPoints: teamPoints,
            opponentPoints: opponentPoints
        }
        this._games.push(game)
    }
};

team.addPlayer('Bugs', 'Bunny', 76);
console.log(team.players);
team.addGame('Titans', 100, 98);
console.log(team.games);
