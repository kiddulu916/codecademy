# Import the robot race functions and other useful libraries
import robot_race_functions as rr
from collections import deque, Counter, namedtuple
from time import time, sleep

# Define maze file that will be used
maze_file_name = 'maze_data_1.csv'

# Set the seconds between each turn and the max number of turns that the race will end at
seconds_between_turns = 0.3
max_turns = 35

# Initialize the robot race
maze_data = rr.read_maze(maze_file_name)
rr.print_maze(maze_data)
walls, goal, bots = rr.process_maze_init(maze_data)

# Populate a deque of all robot commands for the provided maze
robot_moves = deque()
num_of_turns = 0
while not rr.is_race_over(bots) and num_of_turns < max_turns:
  # For each bot, if the bot has not reached the goal, add a new move to robot_moves deque
  for bot in bots:
    if bot.has_finished == False:
      robot_moves.append(rr.compute_robot_logic(walls, goal, bot))
  num_of_turns += 1
  
move_count = Counter(move[0] for move in robot_moves)
collision_count = Counter(move[0] for move in robot_moves if move[2] == True)

BotScoreData = namedtuple('BotScoreData', ['name', 'num_moves', 'num_collisions', 'score'])

bot_scores = []
for bot in bots:
  bot_scores.append(BotScoreData(bot.name, move_count[bot.name], collision_count[bot.name], move_count[bot.name] + collision_count[bot.name]))

bot_data = {}
for bot in bots:
  bot_data[bot.name] = bot
  
while len(robot_moves) > 0:
  bot_name, direction, has_collided = robot_moves.popleft()
  bot_data[bot_name].process_move(direction)
  rr.update_maze_characters(maze_data, bots)
  rr.print_maze(maze_data)
  sleep(seconds_between_turns - time() % seconds_between_turns)
  
rr.print_results(bot_scores)
  