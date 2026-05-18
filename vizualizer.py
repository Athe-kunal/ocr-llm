from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDFS_ROOT = ROOT / "pdfs"
MARKDOWN_ROOT = ROOT / "pdf_dir" / "markdown" / "pdfs"
PAGE_RE = re.compile(
    r"<PAGE-NUM-(?P<num>\d+)>(?P<body>.*?)</PAGE-NUM-(?P=num)>",
    re.DOTALL,
)


@dataclass(frozen=True)
class PageBlock:
    page_num: int
    body: str
    start: int
    end: int


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def folder_options() -> list[str]:
    folders = {path.relative_to(MARKDOWN_ROOT).as_posix() for path in MARKDOWN_ROOT.iterdir() if path.is_dir()}
    if PDFS_ROOT.exists():
        folders.update(path.relative_to(PDFS_ROOT).as_posix() for path in PDFS_ROOT.iterdir() if path.is_dir())
    return sorted(folders)


def safe_folder(folder: str) -> str:
    path = Path(folder)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Folder must be a relative path below pdfs/")
    return path.as_posix().strip("/")


def markdown_dir_for(pdf_folder: str) -> Path:
    return MARKDOWN_ROOT / safe_folder(pdf_folder)


def pdf_dir_for(pdf_folder: str) -> Path:
    return PDFS_ROOT / safe_folder(pdf_folder)


def document_options(markdown_dir: Path, pdf_dir: Path) -> list[str]:
    stems = set()
    if markdown_dir.exists():
        stems.update(path.stem for path in markdown_dir.glob("*.md"))
    if pdf_dir.exists():
        stems.update(path.stem for path in pdf_dir.glob("*.pdf"))
    return sorted(stems)


def read_page_blocks(markdown_path: Path) -> list[PageBlock]:
    text = markdown_path.read_text(encoding="utf-8")
    return [
        PageBlock(
            page_num=int(match.group("num")),
            body=match.group("body"),
            start=match.start(),
            end=match.end(),
        )
        for match in PAGE_RE.finditer(text)
    ]


def page_block(markdown_path: Path, page_num: int) -> PageBlock | None:
    for block in read_page_blocks(markdown_path):
        if block.page_num == page_num:
            return block
    return None


def replace_page(markdown_path: Path, page_num: int, new_body: str) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    found = False

    def replacement(match: re.Match[str]) -> str:
        nonlocal found
        if int(match.group("num")) != page_num:
            return match.group(0)
        found = True
        return f"<PAGE-NUM-{page_num}>{new_body}</PAGE-NUM-{page_num}>"

    updated = PAGE_RE.sub(replacement, text)
    if not found:
        raise ValueError(f"PAGE-NUM-{page_num} was not found in {display_path(markdown_path)}")
    markdown_path.write_text(updated, encoding="utf-8")


@st.cache_data(show_spinner=False)
def pdf_page_count(pdf_path_str: str) -> int | None:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return None
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def pdf_page_png(pdf_path_str: str, page_num: int, zoom: float) -> bytes | None:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return None
    with fitz.open(pdf_path) as doc:
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc.load_page(page_num - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


def render_pdf(pdf_path: Path, page_num: int, zoom: float) -> None:
    image = pdf_page_png(str(pdf_path), page_num, zoom)
    if image is None:
        st.info(f"No matching PDF found at `{display_path(pdf_path)}`.")
        return
    st.image(image, use_container_width=True)


def sync_editor(markdown_path: Path, page_num: int) -> None:
    editor_id = f"{markdown_path}:{page_num}"
    if st.session_state.get("editor_id") == editor_id:
        return
    block = page_block(markdown_path, page_num)
    st.session_state.editor_id = editor_id
    st.session_state.page_source = "" if block is None else block.body


def main() -> None:
    st.set_page_config(page_title="OCR PDF/HTML Visualizer", layout="wide")
    st.title("OCR PDF/HTML Visualizer")

    folders = folder_options()
    default_folder = "AAPL-2024" if "AAPL-2024" in folders else (folders[0] if folders else "AAPL-2024")

    with st.sidebar:
        st.header("Document")
        folder = st.text_input("PDF folder", value=default_folder, help="Example: pdfs/AAPL-2024 or AAPL-2024")
        folder = folder.removeprefix("pdfs/").strip("/")

        try:
            markdown_dir = markdown_dir_for(folder)
            pdf_dir = pdf_dir_for(folder)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.caption(f"Markdown: `{display_path(markdown_dir)}`")
        st.caption(f"PDFs: `{display_path(pdf_dir)}`")

        docs = document_options(markdown_dir, pdf_dir)
        if not docs:
            st.error("No matching `.md` or `.pdf` documents found.")
            st.stop()

        doc = st.selectbox("Filing", docs)
        markdown_path = markdown_dir / f"{doc}.md"
        pdf_path = pdf_dir / f"{doc}.pdf"

        if not markdown_path.exists():
            st.error(f"No markdown file found at `{display_path(markdown_path)}`.")
            st.stop()

        blocks = read_page_blocks(markdown_path)
        if not blocks:
            st.error("No `<PAGE-NUM-{n}>...</PAGE-NUM-{n}>` blocks found in the markdown file.")
            st.stop()

        markdown_pages = [block.page_num for block in blocks]
        pdf_pages = pdf_page_count(str(pdf_path))
        max_page = max(markdown_pages + ([pdf_pages] if pdf_pages else []))
        page_state_key = f"page_num:{folder}:{doc}"
        page_input_key = f"page_input:{folder}:{doc}"
        if page_state_key not in st.session_state:
            st.session_state[page_state_key] = markdown_pages[0]
        st.session_state[page_state_key] = min(max(1, st.session_state[page_state_key]), max_page)
        st.session_state[page_input_key] = st.session_state[page_state_key]

        height = st.slider("Viewer height", min_value=400, max_value=1200, value=850, step=50)
        pdf_zoom = st.slider("PDF render zoom", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
        st.caption(f"Markdown pages: {min(markdown_pages)}-{max(markdown_pages)}")
        if pdf_pages:
            st.caption(f"PDF pages: 1-{pdf_pages}")

    def change_page(delta: int) -> None:
        next_page = min(max(1, int(st.session_state[page_state_key]) + delta), max_page)
        st.session_state[page_state_key] = next_page
        st.session_state[page_input_key] = next_page

    def sync_page_input() -> None:
        st.session_state[page_state_key] = int(st.session_state[page_input_key])

    prev_col, page_col, next_col, spacer_col = st.columns([1, 1, 1, 3])
    with prev_col:
        st.button(
            "Previous page",
            use_container_width=True,
            disabled=st.session_state[page_state_key] <= 1,
            on_click=change_page,
            args=(-1,),
        )
    with page_col:
        page_num = st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            key=page_input_key,
            step=1,
            label_visibility="collapsed",
            on_change=sync_page_input,
        )
        page_num = int(page_num)
    with next_col:
        st.button(
            "Next page",
            use_container_width=True,
            disabled=st.session_state[page_state_key] >= max_page,
            on_click=change_page,
            args=(1,),
        )
    with spacer_col:
        st.caption(f"Page {page_num} of {max_page}")

    sync_editor(markdown_path, page_num)
    block = page_block(markdown_path, page_num)
    if block is None:
        st.warning(f"`PAGE-NUM-{page_num}` does not exist in `{display_path(markdown_path)}`.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("PDF Page")
        render_pdf(pdf_path, page_num, pdf_zoom)

    with right:
        st.subheader("Editable Source")
        edited = st.text_area(
            "Edit page source",
            key="page_source",
            height=height,
            help="HTML/markdown inside the selected PAGE-NUM block",
        )
        save_col, reload_col = st.columns(2)
        with save_col:
            if st.button("Save page", type="primary", use_container_width=True, disabled=block is None):
                replace_page(markdown_path, page_num, edited)
                st.cache_data.clear()
                st.success(f"Saved page {page_num}.")
        with reload_col:
            if st.button("Reload page", use_container_width=True):
                st.session_state.pop("editor_id", None)
                sync_editor(markdown_path, page_num)
                st.rerun()
        st.caption(f"Editing `{display_path(markdown_path)}`")


if __name__ == "__main__":
    main()
