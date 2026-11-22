import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if any(arg in sys.argv[1:] for arg in ('--check', '--ocr', '--download')):
        from farsi_ocr.cli import main as cli_main
        sys.exit(cli_main(sys.argv[1:]))
    from farsi_ocr.gui import main
    sys.exit(main())
