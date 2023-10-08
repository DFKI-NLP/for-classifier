import sparql_dataframe

def get_academicDisciplines():

    endpoint = "http://dbpedia.org/sparql"

    query = """
        PREFIX :     <http://dbpedia.org/resource/>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo:  <http://dbpedia.org/ontology/>

        SELECT DISTINCT ?discipline ?label ?abstract

        WHERE {
        ?subject dbo:academicDiscipline ?discipline .
        ?discipline rdfs:label ?label ;
                        dbo:abstract ?abstract .
        FILTER (LANG(?label)="en") .
        FILTER (LANG(?abstract)="en") .
        }
    """

    # use sparql_dataframe; a library that return results of SPARQL queries as pandas DataFrames
    academicDisciplines = sparql_dataframe.get(endpoint, query)

    return academicDisciplines



