import great_expectations as gx

context = gx.get_context()

# Add Pandas runtime datasource (Fluent API)
if "pandas_datasource" not in [ds.name for ds in context.list_datasources()]:
    context.sources.add_pandas(
        name="pandas_datasource"
    )

print("✅ Pandas datasource ready for Fluent API!")