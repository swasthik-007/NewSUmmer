function getqueryparams(){
    const queryparams=new URLSearchParams(window.location.search);
       const queryparamsobject=Object.fromEntries(queryparams.entries());
    return queryparamsobject;
    }

    async function fetchproductbyid(id){
        const product =await axios.get(`https://fakestoreapi.com/products/${id}`);
        return product.data;
    }

    function removeloader(){
        const loaderbackdrop=document.getElementById("loaderbackdrop");
        loaderbackdrop.style.display = 'none';
    }
    async function fetchcardsbyid(id){
        const cart = await axios.get(`https://fakestoreapi.com/carts/${id}`);
        return cart.data;
    }