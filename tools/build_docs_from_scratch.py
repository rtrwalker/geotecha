"""delete "..\docs\_build\" directory and build html docs"""


from __future__ import print_function, division

import _docs_delete__build_dir
import _docs_html_docs


def main():
    """run _docs_delete__build_dir, _docs_html_docs, """


    print('build docs from scratch')
    _docs_delete__build_dir.main()
    _docs_html_docs.main()


if __name__ == '__main__':
    main()
