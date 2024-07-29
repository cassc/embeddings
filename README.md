
Build embeddings database using function source code in the function database

# Requirement

- Smart contract database `ipfs://QmZistNGreYdKMXDVhrFmdynXTyiaTTvTh4915MvtkvoUR`  from https://github.com/sbip-sg/blockchain-data
- A model for generating embeddings, for example https://huggingface.co/bigcode/starencoder

# Usage

```bash
# Build embeddings database:
python lib/build-embedding.py --model-name /data-ssd/chen/starencoder/ --embeddings-db-path /data-ssd/chen/embeddings.duckdb --function-db-path /data-ssd/chen/contracts.duckdb


# Find similar functions:
python lib/use-embedding.py --model-name /data-ssd/chen/starencoder/ --embeddings-db-path /data-ssd/chen/embeddings.0729.duckdb --function-db-path /data-ssd/chen/contracts.duckdb --limit 10 \
   ' function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external virtual override payable ensure(deadline) returns (uint amountToken, uint amountETH, uint liquidity) {
        (amountToken, amountETH) = _addLiquidity(
            token,
            WETH,
            amountTokenDesired,
            msg.value,
            amountTokenMin,
            amountETHMin
        );
        address pair = UniswapV2Library.pairFor(factory, token, WETH);
        TransferHelper.safeTransferFrom(token, msg.sender, pair, amountToken);
        IWETH(WETH).deposit{value: amountETH}();
        assert(IWETH(WETH).transfer(pair, amountETH));
        liquidity = IUniswapV2Pair(pair).mint(to);
        // refund dust eth, if any
        if (msg.value > amountETH) TransferHelper.safeTransferETH(msg.sender, msg.value - amountETH);
    }
    '
```

# Limitations

The source code will be truncated to `--max-length` characters (default 1024) if it contains more than that. This limit is imposed by the model used to generate embeddings.
