from PyPDF2 import PdfReader
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import TokenTextSplitter

async def get_chunks(files):
    node_parser = TokenTextSplitter(chunk_size=256, chunk_overlap=128)
    nodes = []
    for file in files:
        file_content = await file.read()
        nodes.extend(node_parser.get_nodes_from_documents([Document(text=file_content)], show_progress=False))
    return nodes


async def get_pdf_text(file):
    reader = PdfReader(file)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text


async def get_chunks_pdf(files):
    node_parser = TokenTextSplitter(chunk_size=256, chunk_overlap=128)
    nodes = []
    for file in files:
        file_content = await get_pdf_text(file.file)
        nodes.extend(node_parser.get_nodes_from_documents([Document(text=file_content)], show_progress=False))
    return nodes

