# traj1 (내자리 -> 프린터기)
- low-level instruction
```
python3 online_eval_uninavid.py --instruction "Right now you are standing in front of a blue door. Walk straight towards that door. When you get very close to the fire hydrant, turn right. You will see a gray trash can. Turn right again until you can fully see the refrigerator in front of you. Then, walk straight towards the printer paper. That is your destination" --save_video
```
- human-like instruction
```
python3 online_eval_uninavid.py --instruction "Go straight, turn right, and head toward the printer paper. That's your goal." --save_video
```

# traj2 (연구실 문앞 -> 화장실 옆쪽 파란문)
- low-level instruction

```
python3 online_eval_uninavid.py --instruction "You are standing facing the door at the end of the hallway. Walk straight towards that door until you get close to it. When you see a new path on your left, continue walking straight until you are about halfway past that left path, then turn right. You will see a new blue door. Walk straight towards it. That is your destination." --save_video
```
- human-like instruction 
```
python3 online_eval_uninavid.py --instruction "Go straight and make a left. You'll see a blue door right ahead—that's your destination." --save_video
```

# traj3 (비상구 오른쪽 -> 왼쪽 캐비넷)
- low-level instruction
```
python3 online_eval_uninavid.py --instruction "Right now you are facing a blue door and a trash can. Walk straight ahead. When you get close to the trash can, turn left. In the new hallway, if you see a cabinet, walk straight towards it. That is your destination." --save_video
```
- human-like instruction
```
python3 online_eval_uninavid.py --instruction "Head straight and turn left. You'll spot a cabinet, and that's the goal." --save_video
```

# traj4 (화장실 앞 -> 베란다 입구)
- low-level instruction
```
python3 online_eval_uninavid.py --instruction "You are standing in a hallway facing a glass railing. Walk slightly forward towards the glass railing. When you get close to the trash can, turn right. You will see a new hallway. Walk straight down this hallway. As you approach the end of the hallway, turn left and walk straight towards the door in front of you. That door is your destination." --save_video
```
- human-like instruction 
```
# traj4 (human-like instruction)
python3 online_eval_uninavid.py --instruction "Go straight, turn right, and keep going. Just head out the door on your left, and that's the goal." --save_video
```

# traj5 (내자리에서 앞에 문 열고 -> 비상구)
- low-level instruction
```
python3 online_eval_uninavid.py --instruction "You are standing facing an open door. Walk straight and once you fully exit through the door, turn right. You will see a long hallway. Walk straight until you see a new path on your left. Walk a little further past the path on the left, then turn left to enter that hallway. You will see a hallway with a blue door at the end. Walk slightly forward towards it, then turn left into the emergency exit area on your left and enter. That is your destination." --save_video
```
- human-like instruction
```
python3 online_eval_uninavid.py --instruction "Head out, turn right, and go straight. Take a left at the hallway and keep walking. You'll see an emergency exit on your left—just go through there, and that's your destination." --save_video
```
