from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "The old library stood at the corner of Main Street, its red brick facade weathered by decades of rain and sun. Inside, the smell of aged paper and leather bindings filled the air, creating an atmosphere that transported visitors to another era. Rows upon rows of wooden shelves stretched toward the high ceiling, each one packed with books that held stories, knowledge, and forgotten memories. A librarian sat behind the front desk, her reading glasses perched on her nose as she carefully stamped the due date on a patron's borrowed books. Near the window, a young student hunched over a thick textbook, occasionally glancing outside at the world passing by. The afternoon sunlight streamed through the tall windows, casting long shadows across the polished wooden floor. In the children's section, colorful posters decorated the walls, and small chairs surrounded low tables covered with picture books. An elderly man dozed in a leather armchair, his newspaper slipping from his grasp. The clock on the wall ticked steadily, marking the passage of time in this sanctuary of silence and learning. Outside, the modern world rushed by, but within these walls, time seemed to move at its own gentle pace."


splitter = RecursiveCharacterTextSplitter(chunk_size= 50, chunk_overlap = 10)

chunks = splitter.split_text(text)
total_chunks= len(chunks)


for i,chunk in enumerate (chunks,1):
    chunk_characters = len(chunk)
    print(f"Chunk {i} ({chunk_characters}): {chunk}")





