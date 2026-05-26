import pdfplumber


class TableParser:

    def extract_tables(self, path):
        tables = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_table()
                if extracted:
                    tables.append(extracted)

        return tables