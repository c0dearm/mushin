use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, Data, DeriveInput, Expr, ExprLit,
    Field, Fields, GenericArgument, Ident, Lit, Path, PathArguments::AngleBracketed, Type,
    TypePath,
};

#[proc_macro_derive(NeuralNetwork)]
pub fn derive_neural_network(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            Fields::Unnamed(_) | Fields::Unit => unimplemented!(),
        },
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    };

    proc_macro::TokenStream::from(impl_neural_network(name, fields))
}

fn get_field_type_args(field: &Field) -> &Punctuated<GenericArgument, Comma> {
    let type_args = &match &field.ty {
        Type::Path(TypePath {
            qself: _,
            path: Path {
                leading_colon: _,
                segments,
            },
        }) => segments,
        _ => unimplemented!(),
    }[0]
    .arguments;

    match type_args {
        AngleBracketed(args) => &args.args,
        _ => unimplemented!(),
    }
}

fn as_usize(arg: &GenericArgument) -> usize {
    match arg {
        GenericArgument::Const(Expr::Lit(ExprLit {
            attrs: _,
            lit: Lit::Int(v),
        })) => v.base10_parse::<usize>().unwrap(),
        _ => unimplemented!(),
    }
}

fn impl_neural_network(name: Ident, fields: Punctuated<Field, Comma>) -> TokenStream {
    let forward_chain = fields.iter().fold(quote!(input), |acc, f| {
        let name = &f.ident;
        quote!(self.#name.forward(#acc))
    });

    let input_size = as_usize(&get_field_type_args(fields.first().unwrap())[1]);
    let output_size = as_usize(&get_field_type_args(fields.last().unwrap())[2]);

    quote! {
        impl NeuralNetwork<#input_size, #output_size> for #name {
            fn forward(&self, input: [f32; #input_size]) -> [f32; #output_size] {
                #forward_chain
            }
        }
    }
}
