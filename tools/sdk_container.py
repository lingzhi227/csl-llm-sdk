"""Explicit SDK container mount ordering with task-owned temporary storage."""
import argparse
import os
from pathlib import Path


def command(image, work, argv):
    work, image = Path(work).resolve(), Path(image).resolve()
    if not work.is_dir() or not image.is_file():
        raise ValueError('Existing work directory and SDK image required')
    temporary = work/'tmp'
    temporary.mkdir(exist_ok=True)
    return ['singularity', 'exec', '-C', '--bind='+str(work)+':'+str(work),
            '--bind='+str(temporary)+':/tmp', '--pwd='+str(work),
            '--env=TMPDIR=/tmp', str(image), *argv]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv:
        parser.error('Container command required after --')
    cmd = command(args.image, Path.cwd(), argv)
    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
