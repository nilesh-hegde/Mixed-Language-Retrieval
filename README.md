# Mixed-Language-Retrieval

Mixed-Language Information Retrieval is concerned with retrieval from a document collection where documents in multiple languages co-exist and need to be retrieved to a query in any language. In multicultural and multilingual environments, users often speak and read multiple languages. They may be more comfortable or proficient in one language for certain tasks and another language for different tasks. Multi-language retrieval allows users to find information in the languages they are most comfortable with. Also, with the drastic increase of content on the internet, there is a wealth of information available in multiple languages. Multi-language retrieval enables users to access this diverse content and find relevant information regardless of the language in which it’s written.

In this repository, the queries used in the mixed-language retrieval system contain a mixture of three languages: English, Spanish and French. The dataset used in the project is MIRACL (Multilingual Information Retrieval Across a Continuum of Languages), which was an WSDM 2023 Cup challengethat focused on search across 18 different languages,which collectively encompassed over three billion native speakers around the world. That datasetconsists of passages from Wikipedia for the 18 different languages. Searching was done by developing retrieval models such as Okapi BM25, Okapi+TF-IDF and Query-likelihood.

A high level diagram of the system is as follows:

<center>![pipeline](https://github.com/nilesh-hegde/Mixed-Language-Retrieval/assets/55364143/c5701004-04e5-494e-8db2-f7ee8f8465cb)</center>



