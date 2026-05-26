class TableChunker:

    def chunk_tables(self, tables):

        chunks = []

        for table in tables:

            rows = []

            for row in table:
                rows.append(" | ".join([str(cell) for cell in row]))

            table_text = "\n".join(rows)

            chunks.append(table_text)

        return chunks