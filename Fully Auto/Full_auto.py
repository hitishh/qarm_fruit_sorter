import os
import sys
import numpy as np
import time
import cv2

# Add QArm library paths
sys.path.append(r"C:\Users\hitis\Documents\Quanser\0_libraries\python")
sys.path.append(r"C:\Users\hitis\Documents\Quanser\5_research\qarm\basic")

from pal.products.qarm import QArm, QArmRealSense
from hal.products.qarm import QArmUtilities

# Import your classification module
from fruit_ident import classify_and_log

def check_ik_solution_validity(phiCmd):
    return not np.any(np.isnan(phiCmd))

def move_to_position(myArm, myArmUtilities, position, gamma, gripCmd, led_cmd, label):
    _, phiCmd = myArmUtilities.qarm_inverse_kinematics(position, gamma, myArm.measJointPosition[0:4])
    if not check_ik_solution_validity(phiCmd):
        print(f"Unreachable position: {label}")
        return None
    myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=led_cmd)
    time.sleep(2)
    print(f"Moved to {label} position.")
    return phiCmd

def main():
    led_cmd = np.array([0, 1, 0], dtype=np.float64)
    np.set_printoptions(precision=2, suppress=True)

    print("--- Full Auto: Camera or Upload → Classify → Pick → Sort → Repeat ---\n")

    positions = {
        "home": [0.40, 0, 0.30],
        "Pick_pose": [0.65, 0, 0.30],
        "R_tom": [-0.25, -0.4, 0.05],
        "R_ban": [-0.2, -0.5, 0.05],
        "R_str": [-0.3, -0.3, 0.05],
        "U_tom": [-0.35, -0.1, 0.05],
        "U_ban": [-0.42, 0.1, 0.05],
        "U_str": [-0.4, 0.3, 0.05],
        "Rott": [-0.1, 0.4, 0.05]
    }

    counters = {
        "banana_ripe": 0, "banana_unripe": 0,
        "tomato_ripe": 0, "tomato_unripe": 0,
        "strawberry_ripe": 0, "strawberry_unripe": 0
    }

    thresholds = {
        "banana_ripe": 3, "banana_unripe": 3,
        "tomato_ripe": 5, "tomato_unripe": 5,
        "strawberry_ripe": 6, "strawberry_unripe": 6
    }

    with QArm(hardware=0) as myArm, QArmRealSense(
        mode='RGB&DEPTH',
        hardware=0,
        deviceID=0,
        frameWidthRGB=640,
        frameHeightRGB=480,
        frameWidthDepth=640,
        frameHeightDepth=480,
        readMode=0
    ) as myCam:

        myArmUtilities = QArmUtilities()
        gamma = 0

        while True:
            gripCmd = 0.0

            # Step 1: Move to Home
            phiCmd = move_to_position(myArm, myArmUtilities, np.array(positions["home"]), gamma, gripCmd, led_cmd, "HOME")
            if phiCmd is None: break

            # Step 2: Ask for image source
            method = input("Use camera or upload image? (camera/upload): ").strip().lower()
            if method == "camera":
                print("Capturing image from RealSense...")
                myCam.read_RGB()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                rgb_filename = f"rgb_{timestamp}.png"
                cv2.imwrite(rgb_filename, myCam.imageBufferRGB)
                print(f"Saved RGB image to: {rgb_filename}")
            elif method == "upload":
                rgb_filename = input("Enter full image path: ").strip()
                if not os.path.isfile(rgb_filename):
                    print("File not found. Try again.")
                    continue
            else:
                print("Invalid input. Choose 'camera' or 'upload'.")
                continue

            # Step 3: Classify fruit
            fruit, quality = classify_and_log(rgb_filename)
            print(f"Identified: {fruit}, Quality: {quality}")

            # Step 3.1: Update and check counters
            key = f"{fruit}_{quality}"
            if key in counters:
                counters[key] += 1
                print(f"Counter [{key}]: {counters[key]} / {thresholds[key]}")
                if counters[key] >= thresholds[key]:
                    print(f"Move {quality} {fruit} conveyor by 1")
                    counters[key] = 0

            # Step 4: Move to pick and close gripper
            phiCmd = move_to_position(myArm, myArmUtilities, np.array(positions["Pick_pose"]), gamma, gripCmd, led_cmd, "PICK")
            if phiCmd is None: break

            gripCmd = 1.0
            myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=led_cmd)
            print("Gripper closed at PICK position.")
            time.sleep(1)

            # Step 5: Decide destination
            destination_key = ""
            if quality == "ripe":
                if fruit == "banana":
                    destination_key = "R_ban"
                elif fruit == "tomato":
                    destination_key = "R_tom"
                elif fruit == "strawberry":
                    destination_key = "R_str"
            elif quality == "unripe":
                if fruit == "banana":
                    destination_key = "U_ban"
                elif fruit == "tomato":
                    destination_key = "U_tom"
                elif fruit == "strawberry":
                    destination_key = "U_str"
            elif quality == "rotten":
                destination_key = "Rott"
            else:
                print("Unknown classification result.")
                break

            # Step 6: Move to destination and release
            if destination_key in positions:
                phiCmd = move_to_position(myArm, myArmUtilities, np.array(positions[destination_key]), gamma, gripCmd, led_cmd, f"PLACE - {destination_key}")
                if phiCmd is None: break

                gripCmd = 0.0
                myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=led_cmd)
                print("Gripper opened to place item.")
                time.sleep(1)
            else:
                print("No matching destination. Skipping place.")
                break

            # Step 7: Return to home
            phiCmd = move_to_position(myArm, myArmUtilities, np.array(positions["home"]), gamma, gripCmd, led_cmd, "HOME")
            if phiCmd is None: break

            # Step 8: Ask user to continue or not
            user_input = input("\nContinue to next fruit? (yes/no): ").strip().lower()
            if user_input != "yes":
                print("Terminating after this cycle.")
                break

    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()
