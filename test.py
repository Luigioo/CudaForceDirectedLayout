import subprocess

# Call the compiled C++ program
result = subprocess.run(["./x64/Debug/CudaRuntime2.exe"], capture_output=True, text=True)

# Check the return code
if result.returncode == 0:
    # Print the program output
    print("Program output:\n", result.stdout)
else:
    # Print the error message
    print("An error occurred:\n", result.stderr)