- robot_control
```
lrclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
root@spark-c9f0:/when2reason/Uni-NaVid# python robot_control.py 
[INFO] [1768458925.496307548] [robot_action_controller]: Uni-NaVid Optimized Controller Ready.
[INFO] [1768459000.028008653] [robot_action_controller]: Action [1/4]: forward
[INFO] [1768459000.028232099] [robot_action_controller]: Queue Updated: ['forward', 'forward', 'forward', 'forward']
[INFO] [1768459000.757159464] [robot_action_controller]: Queue Updated: ['left', 'forward', 'forward', 'forward']
[INFO] [1768459000.757878050] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459001.359696158] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'forward']
[INFO] [1768459001.359827060] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459001.764138554] [robot_action_controller]: Queue Updated: ['right', 'forward', 'forward', 'right']
[INFO] [1768459001.888073495] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459002.223025011] [robot_action_controller]: Queue Updated: ['right', 'right', 'forward', 'forward']
[INFO] [1768459002.488186570] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459002.925906207] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459002.988873894] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459003.142974233] [robot_action_controller]: Queue Updated: ['right', 'right', 'forward', 'forward']
[INFO] [1768459003.594748320] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459003.654085609] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459003.881077289] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459004.188030241] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459004.359394611] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459004.674669258] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459004.788014602] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459005.279255450] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'forward']
[INFO] [1768459005.387741007] [robot_action_controller]: Action [1/4]: right
[INFO] [1768459005.686232963] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459005.834243758] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459005.887848195] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459006.198812387] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459006.201197020] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459006.388040274] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459006.519243714] [robot_action_controller]: Queue Updated: ['right', 'right', 'right', 'right']
[INFO] [1768459006.719777311] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459006.987944045] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459007.488375181] [robot_action_controller]: Action [2/4]: left
[INFO] [1768459008.087924912] [robot_action_controller]: Action [3/4]: left
[INFO] [1768459008.969696167] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459008.975147267] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459009.487894017] [robot_action_controller]: Action [2/4]: left
[INFO] [1768459010.025703967] [robot_action_controller]: Action [1/4]: left
[INFO] [1768459010.025940803] [robot_action_controller]: Queue Updated: ['left', 'left', 'left', 'left']
[INFO] [1768459010.587813372] [robot_action_controller]: Action [2/4]: left
^CTraceback (most recent call last):
  File "/when2reason/Uni-NaVid/robot_control.py", line 231, in <module>

```


- inference
```
Press Ctrl+C to stop

Step 1 | Inf: 0.263s | Comm: 0.000s | Actions: ['forward', 'forward', 'forward', 'forward'] | Published
Step 2 | Inf: 0.162s | Comm: 0.005s | Actions: ['forward', 'forward', 'forward', 'forward'] | Skipped (duplicate)
Step 3 | Inf: 0.158s | Comm: 0.172s | Actions: ['forward', 'forward', 'forward', 'forward'] | Skipped (duplicate)
Step 4 | Inf: 0.160s | Comm: 0.341s | Actions: ['forward', 'forward', 'forward', 'right'] | Published
Step 5 | Inf: 0.159s | Comm: 0.005s | Actions: ['forward', 'forward', 'forward', 'forward'] | Published
Step 6 | Inf: 0.160s | Comm: 0.005s | Actions: ['left', 'forward', 'forward', 'forward'] | Published
Step 7 | Inf: 0.162s | Comm: 0.005s | Actions: ['forward', 'forward', 'right', 'right'] | Published
Step 8 | Inf: 0.161s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 9 | Inf: 0.163s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 10 | Inf: 0.161s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 11 | Inf: 0.161s | Comm: 0.005s | Actions: ['right', 'forward', 'forward', 'forward'] | Published
Step 12 | Inf: 0.162s | Comm: 0.005s | Actions: ['right', 'forward', 'forward', 'right'] | Published
Step 13 | Inf: 0.162s | Comm: 0.005s | Actions: ['right', 'forward', 'forward', 'forward'] | Published
Step 14 | Inf: 0.174s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 15 | Inf: 0.177s | Comm: 0.005s | Actions: ['right', 'right', 'forward', 'forward'] | Published
Step 16 | Inf: 0.177s | Comm: 0.005s | Actions: ['right', 'right', 'forward', 'forward'] | Skipped (duplicate)
Step 17 | Inf: 0.178s | Comm: 0.188s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 18 | Inf: 0.176s | Comm: 0.005s | Actions: ['right', 'right', 'forward', 'forward'] | Published
Step 19 | Inf: 0.179s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 20 | Inf: 0.177s | Comm: 0.005s | Actions: ['right', 'right', 'forward', 'forward'] | Published
Step 21 | Inf: 0.169s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 22 | Inf: 0.169s | Comm: 0.005s | Actions: ['right', 'right', 'forward', 'forward'] | Published
Step 23 | Inf: 0.174s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 24 | Inf: 0.173s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 25 | Inf: 0.174s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Skipped (duplicate)
Step 26 | Inf: 0.173s | Comm: 0.185s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 27 | Inf: 0.179s | Comm: 0.006s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 28 | Inf: 0.177s | Comm: 0.191s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 29 | Inf: 0.180s | Comm: 0.006s | Actions: ['right', 'right', 'right', 'right'] | Skipped (duplicate)
Step 30 | Inf: 0.179s | Comm: 0.192s | Actions: ['right', 'right', 'right', 'right'] | Skipped (duplicate)
Step 31 | Inf: 0.180s | Comm: 0.376s | Actions: ['right', 'right', 'right', 'right'] | Skipped (duplicate)
Step 32 | Inf: 0.180s | Comm: 0.562s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 33 | Inf: 0.179s | Comm: 0.011s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 34 | Inf: 0.180s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Skipped (duplicate)
Step 35 | Inf: 0.180s | Comm: 0.190s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 36 | Inf: 0.183s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 37 | Inf: 0.183s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 38 | Inf: 0.183s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 39 | Inf: 0.186s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 40 | Inf: 0.184s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 41 | Inf: 0.184s | Comm: 0.193s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 42 | Inf: 0.186s | Comm: 0.382s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 43 | Inf: 0.186s | Comm: 0.573s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 44 | Inf: 0.188s | Comm: 0.763s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 45 | Inf: 0.186s | Comm: 0.956s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 46 | Inf: 0.188s | Comm: 1.147s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 47 | Inf: 0.184s | Comm: 1.339s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 48 | Inf: 0.189s | Comm: 1.529s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 49 | Inf: 0.187s | Comm: 0.005s | Actions: ['left', 'left', 'forward', 'forward'] | Published
Step 50 | Inf: 0.189s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 51 | Inf: 0.186s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 52 | Inf: 0.190s | Comm: 0.196s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 53 | Inf: 0.204s | Comm: 0.391s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 54 | Inf: 0.203s | Comm: 0.601s | Actions: ['right', 'right', 'right', 'forward'] | Published
Step 55 | Inf: 0.206s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 56 | Inf: 0.207s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 57 | Inf: 0.208s | Comm: 0.218s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 58 | Inf: 0.207s | Comm: 0.431s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 59 | Inf: 0.207s | Comm: 0.644s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 60 | Inf: 0.205s | Comm: 0.857s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 61 | Inf: 0.211s | Comm: 1.069s | Actions: ['left', 'left', 'left', 'forward'] | Published
Step 62 | Inf: 0.207s | Comm: 0.006s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 63 | Inf: 0.207s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'forward'] | Published
Step 64 | Inf: 0.208s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 65 | Inf: 0.207s | Comm: 0.005s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 66 | Inf: 0.207s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 67 | Inf: 0.207s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 68 | Inf: 0.206s | Comm: 0.223s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 69 | Inf: 0.206s | Comm: 0.434s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 70 | Inf: 0.206s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 71 | Inf: 0.209s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 72 | Inf: 0.206s | Comm: 0.219s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 73 | Inf: 0.208s | Comm: 0.431s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 74 | Inf: 0.208s | Comm: 0.644s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)
Step 75 | Inf: 0.205s | Comm: 0.857s | Actions: ['right', 'right', 'right', 'right'] | Published
Step 76 | Inf: 0.208s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Published
Step 77 | Inf: 0.207s | Comm: 0.005s | Actions: ['left', 'left', 'left', 'left'] | Skipped (duplicate)

```
