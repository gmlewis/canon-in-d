import os

# To print all environment variables Blender sees:
for key, value in os.environ.items():
    print(f"{key}={value}")

# Or to check a specific environment variable:
print(os.getenv('VARIABLE_NAME', 'Not set'))
