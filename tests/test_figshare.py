import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from pypillometry.io import (
    _figshare_article, get_figshare_files, fetch_figshare_file,
)

CONTENT = b"some eyetracking data\n"
FILES = [
    {"id": 1, "name": "a.asc", "size": len(CONTENT),
     "download_url": "https://ndownloader.figshare.com/files/1",
     "computed_md5": hashlib.md5(CONTENT).hexdigest()},
    {"id": 2, "name": "b.pypil", "size": 10,
     "download_url": "https://ndownloader.figshare.com/files/2",
     "computed_md5": ""},
]


def _response(json=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json
    resp.raise_for_status.return_value = None
    return resp


def _fake_download(content):
    def download(url, fname, session=None):
        with open(fname, "wb") as f:
            f.write(content)
        return fname
    return download


class TestFigshareArticle(unittest.TestCase):
    def test_id(self):
        self.assertEqual(_figshare_article(28776617), ("28776617", None))
        self.assertEqual(_figshare_article("28776617"), ("28776617", None))

    def test_url(self):
        url = "https://figshare.com/articles/dataset/Example_pypillometry_dataset/28776617"
        self.assertEqual(_figshare_article(url), ("28776617", None))
        self.assertEqual(_figshare_article(url + "/2"), ("28776617", 2))
        self.assertEqual(_figshare_article(url + "?file=53599913"), ("28776617", None))

    def test_doi(self):
        self.assertEqual(_figshare_article("10.6084/m9.figshare.28776617.v1"),
                         ("28776617", 1))
        self.assertEqual(_figshare_article("https://doi.org/10.6084/m9.figshare.28776617"),
                         ("28776617", None))

    def test_invalid(self):
        for bad in ["https://osf.io/abc12", "not an id", "https://figshare.com/"]:
            with self.assertRaises(ValueError):
                _figshare_article(bad)


class TestGetFigshareFiles(unittest.TestCase):
    @patch("requests.get")
    def test_list(self, mock_get):
        mock_get.return_value = _response(FILES)
        files = get_figshare_files(123)
        self.assertEqual(list(files), ["a.asc", "b.pypil"])
        self.assertEqual(files["a.asc"]["id"], 1)
        self.assertEqual(files["a.asc"]["md5"], FILES[0]["computed_md5"])
        url = mock_get.call_args.args[0]
        self.assertEqual(url, "https://api.figshare.com/v2/articles/123/files")

    @patch("requests.get")
    def test_version(self, mock_get):
        mock_get.return_value = _response(FILES)
        get_figshare_files("10.6084/m9.figshare.123.v3")
        self.assertEqual(mock_get.call_args.args[0],
                         "https://api.figshare.com/v2/articles/123/versions/3/files")
        get_figshare_files("10.6084/m9.figshare.123.v3", version=1)
        self.assertEqual(mock_get.call_args.args[0],
                         "https://api.figshare.com/v2/articles/123/versions/1/files")

    @patch("requests.get")
    def test_pagination(self, mock_get):
        page1 = [dict(FILES[0], id=i, name=f"f{i}") for i in range(1000)]
        mock_get.side_effect = [_response(page1), _response(FILES)]
        files = get_figshare_files(123)
        self.assertEqual(len(files), 1002)
        self.assertEqual(mock_get.call_args.kwargs["params"]["page"], 2)

    @patch("requests.get")
    def test_not_found(self, mock_get):
        mock_get.return_value = _response({"code": "EntityNotFound"}, status=404)
        with self.assertRaises(ValueError):
            get_figshare_files(123)


class TestFetchFigshareFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    @patch("pypillometry.io.download", side_effect=_fake_download(CONTENT))
    @patch("requests.get", return_value=_response(FILES))
    def test_download_and_cache(self, mock_get, mock_download):
        fname = fetch_figshare_file(123, "a.asc", cache_dir=self.tmpdir)
        self.assertEqual(fname, os.path.join(self.tmpdir, "a.asc"))
        with open(fname, "rb") as f:
            self.assertEqual(f.read(), CONTENT)
        # cached: no further requests
        fetch_figshare_file(123, "a.asc", cache_dir=self.tmpdir)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_download.call_count, 1)
        fetch_figshare_file(123, "a.asc", cache_dir=self.tmpdir, force_download=True)
        self.assertEqual(mock_download.call_count, 2)

    @patch("pypillometry.io.download", side_effect=_fake_download(CONTENT))
    @patch("requests.get", return_value=_response(FILES))
    def test_mirror_layout(self, mock_get, mock_download):
        fname = fetch_figshare_file(123, "a.asc", version=2, cache_dir=self.tmpdir,
                                    cache_layout="mirror")
        self.assertEqual(fname, os.path.join(self.tmpdir, "figshare.com", "123",
                                             "v2", "a.asc"))

    @patch("pypillometry.io.download", side_effect=_fake_download(b"corrupted"))
    @patch("requests.get", return_value=_response(FILES))
    def test_checksum_mismatch(self, mock_get, mock_download):
        with self.assertRaises(ValueError):
            fetch_figshare_file(123, "a.asc", cache_dir=self.tmpdir)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "a.asc")))

    @patch("requests.get", return_value=_response(FILES))
    def test_unknown_file(self, mock_get):
        with self.assertRaisesRegex(ValueError, "b.pypil"):
            fetch_figshare_file(123, "missing.asc", cache_dir=self.tmpdir)

    def test_bad_name(self):
        for bad in ["../a.asc", "sub/a.asc", ""]:
            with self.assertRaises(ValueError):
                fetch_figshare_file(123, bad, cache_dir=self.tmpdir)


if __name__ == "__main__":
    unittest.main()
