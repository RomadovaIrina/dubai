import sys,subprocess,json
r=subprocess.run(["/venv/mediapipe/bin/python","/workspace/dub/keysync/k_cavity.py","--pairs","x="+sys.argv[1]+":"+sys.argv[2]],capture_output=True,text=True).stdout
x=json.loads(r)["x"]; print(x["black_hole_share"],x["open_frames"],x["cavity_L_mean"],x["teeth_bright_share"])
