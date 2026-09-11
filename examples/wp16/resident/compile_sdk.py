"""Compilation is disabled for the simulation-only continuation."""

def main():
    raise ValueError('SDK002 reuses unchanged SDK001 outputs; compilation is disabled')

if __name__ == '__main__':
    main()
