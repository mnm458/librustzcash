//! The Signer role (capability holders can contribute).
//!
//! - Needs the spend authorization randomizers to create signatures.
//! - Needs sufficient information to verify that the proof is over the correct data,
//!   without needing to verify the proof itself.
//! - A Signer should only need to implement:
//!   - Pedersen commitments using Jubjub / Pallas arithmetic (for note and value
//!     commitments)
//!   - BLAKE2b and BLAKE2s (and the various PRFs / CRHs they are used in)
//!   - Nullifier check (using Jubjub / Pallas arithmetic)
//!   - KDF plus note decryption (AEAD_CHACHA20_POLY1305)
//!   - SignatureHash algorithm
//!   - Signatures (RedJubjub / RedPallas)
//!   - A source of randomness.
//!
//! # Outputs a Signer must tolerate
//!
//! A zero-valued Orchard-protocol output whose `enc_ciphertext` does not decrypt is
//! well-formed. A Signer MUST classify it as a tolerable dummy output, and MUST NOT
//! reconstruct an expected ciphertext and reject the mismatch.
//!
//! After NU6.3 every Orchard-pool action is subject to the same-address restriction, so
//! spending in the Orchard pool pairs each real spend with a fabricated zero-valued
//! output addressed to the spent note's own receiver. That output's `enc_ciphertext` is
//! random bytes rather than an encryption of the note plaintext ([ZIP 326]): a real
//! encryption would trial-decrypt under the receiver's incoming viewing key in the same
//! action that carries the spend's nullifier, letting any holder of that key link the
//! nullifier to the address and so detect the spend.
//!
//! Such an output carries explicit `recipient`, `value`, and `rseed` fields but no
//! `user_address`, so a Signer reads a value of zero and recovers nothing from the
//! ciphertext. The note and its commitment are unaffected, so `cmx` is derived and
//! checked as for any other output.
//!
//! # `user_address` verification
//!
//! An output's `user_address` is an uncommitted display string whose contract is that it
//! contains the output's committed recipient. The sighash commits only to on-wire
//! effects, so a malicious counterparty could otherwise pair an output paying an address
//! they control with a display string the signing user trusts. [`Signer::new`] therefore
//! rejects a PCZT containing any output whose `user_address` is present but does not
//! verifiably contain that output's recipient; see [`UserAddressError`].
//!
//! # Signing before the anchor and witnesses are known
//!
//! For v6 and later transaction formats, a Signer MAY sign a spend whose `witness` is
//! absent, in a bundle whose `anchor` is absent ([ZIP 374]).
//!
//! A v6 txid digest excludes shielded anchors, committing them in the authorizing-data
//! digest instead, and a witness never enters the transaction. The sighash therefore
//! binds every economically meaningful field — nullifiers, value commitments, outputs,
//! and value balances — while the anchor, witnesses, and proofs govern only whether the
//! transaction can be mined, not what it does once it is. They are installed by
//! [`Updater`] before proving.
//!
//! This is what allows a chain of transactions to be signed in one session, before the
//! parent whose outputs they spend has been mined or even broadcast. A Signer's user
//! interface should be able to convey that a batch contains transactions spending the
//! outputs of an in-batch, not-yet-mined parent.
//!
//! For v5 the sighash commits to the anchor, so it must be present before signing.
//!
//! [ZIP 326]: https://zips.z.cash/zip-0326#fabricatedsame-addressoutputsandrandomizednoteciphertexts
//! [ZIP 374]: https://zips.z.cash/zip-0374#anchorsandpre-authorization
//! [`Updater`]: crate::roles::updater::Updater

use alloc::vec::Vec;
use core::convert::Infallible;

use blake2b_simd::Hash as Blake2bHash;
use orchard::primitives::redpallas;
use rand_core::{CryptoRng, Rng};

use ::transparent::{
    address::TransparentAddress,
    sighash::{SIGHASH_ANYONECANPAY, SIGHASH_NONE, SIGHASH_SINGLE, SighashPolicy},
};
use zcash_address::{
    ConversionError, TryFromAddress, ZcashAddress,
    unified::{self, Container, Receiver},
};
use zcash_primitives::transaction::{
    TransactionData, TxDigests, sighash::SignableInput, txid::TxIdDigester,
};
use zcash_protocol::{PoolType, ShieldedPool, consensus::NetworkType};
use zcash_script::script::Evaluable;

use crate::{
    ExtractError, ParsedPczt, Pczt,
    common::{
        FLAG_HAS_SIGHASH_SINGLE, FLAG_SHIELDED_MODIFIABLE, FLAG_TRANSPARENT_INPUTS_MODIFIABLE,
        FLAG_TRANSPARENT_OUTPUTS_MODIFIABLE, Global,
    },
    sighash,
};

pub use crate::EffectsOnly;

pub mod batch;

/// A spend authorization signature for an action in an Orchard-protocol value pool.
///
/// This type only represents signatures for the Orchard and Ironwood value pools; it
/// does not represent Sapling spend authorization signatures. It can be used by external
/// signers that return signatures separately from a PCZT. The value pool and action index
/// identify where the signature belongs when it is later applied with
/// [`Signer::apply_orchard_spend_auth_signature`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpendAuthSignature {
    value_pool: orchard::ValuePool,
    action_index: usize,
    signature: [u8; 64],
}

impl SpendAuthSignature {
    /// Constructs a spend authorization signature for the action at `action_index`
    /// in `value_pool`.
    pub fn from_parts(
        value_pool: orchard::ValuePool,
        action_index: usize,
        signature: [u8; 64],
    ) -> Self {
        Self {
            value_pool,
            action_index,
            signature,
        }
    }

    /// Returns the Orchard-protocol value pool containing the signed action.
    pub fn value_pool(&self) -> orchard::ValuePool {
        self.value_pool
    }

    /// Returns the index of the signed action within its value pool's bundle.
    pub fn action_index(&self) -> usize {
        self.action_index
    }

    /// Returns the raw RedPallas spend authorization signature bytes.
    pub fn signature(&self) -> &[u8; 64] {
        &self.signature
    }
}

/// Extracts the Orchard-protocol spend authorization signatures present in `pczt`.
///
/// The returned signatures are tagged with their Orchard or Ironwood value pool and
/// action index, so they can be transported independently and later applied to a
/// corresponding PCZT with [`Signer::apply_orchard_spend_auth_signature`]. Actions
/// without a spend authorization signature are omitted.
pub fn extract_orchard_spend_auth_signatures(pczt: &Pczt) -> Vec<SpendAuthSignature> {
    fn extract_from_bundle(
        signatures: &mut Vec<SpendAuthSignature>,
        value_pool: orchard::ValuePool,
        bundle: &crate::orchard::Bundle,
    ) {
        for (action_index, action) in bundle.actions().iter().enumerate() {
            if let Some(signature) = action.spend().spend_auth_sig() {
                signatures.push(SpendAuthSignature::from_parts(
                    value_pool,
                    action_index,
                    *signature,
                ));
            }
        }
    }

    let mut signatures = Vec::new();
    extract_from_bundle(&mut signatures, orchard::ValuePool::Orchard, pczt.orchard());
    extract_from_bundle(
        &mut signatures,
        orchard::ValuePool::Ironwood,
        pczt.ironwood(),
    );
    signatures
}

pub struct Signer {
    global: Global,
    transparent: transparent::pczt::Bundle,
    sapling: crate::sapling::Parsed,
    orchard: crate::orchard::Parsed,
    ironwood: crate::orchard::Parsed,
    empty_ironwood: Option<crate::orchard::Bundle>,
    /// Cached across multiple signatures.
    tx_data: TransactionData<EffectsOnly>,
    txid_parts: TxDigests<Blake2bHash>,
    shielded_sighash: [u8; 32],
    transparent_sighash_policy: SighashPolicy,
}

impl Signer {
    /// Instantiates the Signer role with the given PCZT.
    ///
    /// Returns an error if the PCZT's effects cannot be extracted, or if any output's
    /// `user_address` does not verifiably contain that output's committed recipient
    /// (see [`UserAddressError`]).
    pub fn new(pczt: Pczt) -> Result<Self, Error> {
        verify_user_addresses(&pczt)?;

        let anchor_requirement =
            crate::common::AnchorRequirement::for_pre_authorization(pczt.global.tx_version);
        let empty_ironwood = pczt
            .ironwood
            .actions
            .is_empty()
            .then(|| pczt.ironwood.clone());

        let ParsedPczt {
            global,
            transparent,
            sapling,
            orchard,
            ironwood,
            tx_data,
        } = pczt.extract_tx_data(
            anchor_requirement,
            |t| {
                t.extract_effects()
                    .map_err(ExtractError::TransparentExtract)
            },
            |s| s.extract_effects().map_err(ExtractError::SaplingExtract),
            |o| o.extract_effects().map_err(ExtractError::OrchardExtract),
            |i| i.extract_effects().map_err(ExtractError::IronwoodExtract),
        )?;
        let txid_parts = tx_data.digest(TxIdDigester);
        let shielded_sighash = sighash(&tx_data, &SignableInput::Shielded, &txid_parts);

        Ok(Self {
            global,
            transparent,
            sapling,
            orchard,
            ironwood,
            empty_ironwood,
            tx_data,
            txid_parts,
            shielded_sighash,
            transparent_sighash_policy: SighashPolicy::ALL_ONLY,
        })
    }

    /// Sets the sighash policy that this Signer applies to transparent inputs.
    ///
    /// The default is [`SighashPolicy::ALL_ONLY`]; see [`SighashPolicy`] for why the
    /// sighash type a counterparty placed in the PCZT is only used if this Signer’s own
    /// policy permits it.
    pub fn with_transparent_sighash_policy(mut self, sighash_policy: SighashPolicy) -> Self {
        self.transparent_sighash_policy = sighash_policy;
        self
    }

    /// Calculates the signature digest that must be signed to authorize shielded spends.
    ///
    /// This can be used to produce a signature externally suitable for passing to e.g.
    /// [`Self::apply_orchard_signature`].}
    pub fn shielded_sighash(&self) -> [u8; 32] {
        self.shielded_sighash
    }

    /// Calculates the signature digest that must be signed to authorize the transparent
    /// spend at the given index.
    ///
    /// This can be used to produce a signature externally suitable for passing to e.g.
    /// [`Self::append_transparent_signature`].}
    ///
    /// Whoever signs the returned sighash is authorizing whatever it commits to, so the
    /// consistency of the input and the acceptability of its sighash type are checked
    /// here, exactly as they are when this Signer produces the signature itself.
    ///
    /// Returns an error if `index` is invalid for this PCZT.
    pub fn transparent_sighash(&self, index: usize) -> Result<[u8; 32], Error> {
        let input = self
            .transparent
            .inputs()
            .get(index)
            .ok_or(Error::InvalidIndex)?;

        input
            .with_signable_input_with_sighash_policy(
                index,
                |signable_input| {
                    sighash(
                        &self.tx_data,
                        &SignableInput::Transparent(signable_input),
                        &self.txid_parts,
                    )
                },
                self.transparent_sighash_policy,
            )
            .map_err(Error::TransparentSign)
    }

    /// Signs the transparent spend at the given index with the given spending key.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn sign_transparent(
        &mut self,
        index: usize,
        sk: &secp256k1::SecretKey,
    ) -> Result<(), Error> {
        let sighash_policy = self.transparent_sighash_policy;
        self.generate_or_append_transparent_signature(index, |input, tx_data, txid_parts| {
            input.sign_with_sighash_policy(
                index,
                |input| sighash(tx_data, &SignableInput::Transparent(input), txid_parts),
                sk,
                sighash_policy,
            )
        })
    }

    /// Appends the given signature to the transparent spend.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn append_transparent_signature(
        &mut self,
        index: usize,
        signature: secp256k1::ecdsa::Signature,
    ) -> Result<(), Error> {
        let sighash_policy = self.transparent_sighash_policy;
        self.generate_or_append_transparent_signature(index, |input, tx_data, txid_parts| {
            input.append_signature_with_sighash_policy(
                index,
                |input| sighash(tx_data, &SignableInput::Transparent(input), txid_parts),
                signature,
                sighash_policy,
            )
        })
    }

    fn generate_or_append_transparent_signature<F>(
        &mut self,
        index: usize,
        f: F,
    ) -> Result<(), Error>
    where
        F: FnOnce(
            &mut transparent::pczt::Input,
            &TransactionData<EffectsOnly>,
            &TxDigests<Blake2bHash>,
        ) -> Result<(), transparent::pczt::SignerError>,
    {
        let input = self
            .transparent
            .inputs_mut()
            .get_mut(index)
            .ok_or(Error::InvalidIndex)?;

        // The consistency of the input being signed, and the acceptability of its sighash
        // type, are checked by the `zcash_transparent` methods that `f` calls. They are
        // enforced there rather than here because `crate::roles::low_level_signer` hands
        // out the transparent bundle and calls those methods directly.

        // Generate or apply the signature.
        f(input, &self.tx_data, &self.txid_parts).map_err(Error::TransparentSign)?;

        // Update transaction modifiability:
        // - If the Signer added a signature that does not use `SIGHASH_ANYONECANPAY`, the
        //   Transparent Inputs Modifiable Flag must be set to False (because the
        //   signature commits to all inputs, not just the one at `index`).
        if input.sighash_type().encode() & SIGHASH_ANYONECANPAY == 0 {
            self.global.tx_modifiable &= !FLAG_TRANSPARENT_INPUTS_MODIFIABLE;
        }
        // - If the Signer added a signature that does not use `SIGHASH_NONE`, the
        //   Transparent Outputs Modifiable Flag must be set to False. Note that this
        //   applies to `SIGHASH_SINGLE` because we could otherwise remove the output at
        //   `index`, which would not remove the signature.
        if (input.sighash_type().encode() & !SIGHASH_ANYONECANPAY) != SIGHASH_NONE {
            self.global.tx_modifiable &= !FLAG_TRANSPARENT_OUTPUTS_MODIFIABLE;
        }
        // - If the Signer added a signature that uses `SIGHASH_SINGLE`, the Has
        //   `SIGHASH_SINGLE` flag must be set to True.
        if (input.sighash_type().encode() & !SIGHASH_ANYONECANPAY) == SIGHASH_SINGLE {
            self.global.tx_modifiable |= FLAG_HAS_SIGHASH_SINGLE;
        }
        // - Always set the Shielded Modifiable Flag to False.
        self.global.tx_modifiable &= !FLAG_SHIELDED_MODIFIABLE;

        Ok(())
    }

    /// Signs the Sapling spend at the given index with the given spend authorizing key.
    ///
    /// Requires the spend's `proof_generation_key` field to be set (because the API does
    /// not take an FVK).
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    ///
    /// `rng` provides the randomness for the signature.
    pub fn sign_sapling<R: Rng + CryptoRng>(
        &mut self,
        rng: R,
        index: usize,
        ask: &sapling::keys::SpendAuthorizingKey,
    ) -> Result<(), Error> {
        self.generate_or_apply_sapling_signature(index, |spend, shielded_sighash| {
            spend.sign(shielded_sighash, ask, rng)
        })
    }

    /// Applies the given signature to the Sapling spend.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn apply_sapling_signature(
        &mut self,
        index: usize,
        signature: redjubjub::Signature<redjubjub::SpendAuth>,
    ) -> Result<(), Error> {
        self.generate_or_apply_sapling_signature(index, |spend, shielded_sighash| {
            spend.apply_signature(shielded_sighash, signature)
        })
    }

    fn generate_or_apply_sapling_signature<F>(&mut self, index: usize, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut sapling::pczt::Spend, [u8; 32]) -> Result<(), sapling::pczt::SignerError>,
    {
        let spend = self
            .sapling
            .bundle
            .spends_mut()
            .get_mut(index)
            .ok_or(Error::InvalidIndex)?;

        // Check consistency of the input being signed if we have its note components.
        match spend.verify_nullifier(None) {
            Err(
                sapling::pczt::VerifyError::MissingRecipient
                | sapling::pczt::VerifyError::MissingValue
                | sapling::pczt::VerifyError::MissingRandomSeed,
            ) => Ok(()),
            r => r,
        }
        .map_err(Error::SaplingVerify)?;

        // Generate or apply the signature.
        f(spend, self.shielded_sighash).map_err(Error::SaplingSign)?;

        // Update transaction modifiability: all transaction effects have been committed
        // to by the signature.
        self.global.tx_modifiable &= !(FLAG_TRANSPARENT_INPUTS_MODIFIABLE
            | FLAG_TRANSPARENT_OUTPUTS_MODIFIABLE
            | FLAG_SHIELDED_MODIFIABLE);

        Ok(())
    }

    /// Signs the Orchard spend at the given index with the given spend authorizing key.
    ///
    /// Requires the spend's `fvk` field to be set (because the API does not take an FVK).
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    ///
    /// `rng` provides the randomness for the signature.
    pub fn sign_orchard<R: Rng + CryptoRng>(
        &mut self,
        rng: R,
        index: usize,
        ask: &orchard::keys::SpendAuthorizingKey,
    ) -> Result<(), Error> {
        self.generate_or_apply_orchard_signature(index, |spend, shielded_sighash| {
            spend.sign(shielded_sighash, ask, rng)
        })
    }

    /// Applies the given signature to the Orchard spend.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn apply_orchard_signature(
        &mut self,
        index: usize,
        signature: redpallas::Signature<redpallas::SpendAuth>,
    ) -> Result<(), Error> {
        self.generate_or_apply_orchard_signature(index, |action, shielded_sighash| {
            action.apply_signature(shielded_sighash, signature)
        })
    }

    /// Applies an externally produced Orchard-protocol spend authorization signature.
    ///
    /// The signature's value pool selects the Orchard or Ironwood bundle, and its
    /// action index selects the spend within that bundle. The signature is verified
    /// against the action's randomized verification key before it is stored.
    ///
    /// Returns an error if the action index is invalid, the action data is
    /// inconsistent, or the signature does not verify.
    pub fn apply_orchard_spend_auth_signature(
        &mut self,
        signature: &SpendAuthSignature,
    ) -> Result<(), Error> {
        let spend_auth_sig =
            redpallas::Signature::<redpallas::SpendAuth>::from(*signature.signature());
        match signature.value_pool() {
            orchard::ValuePool::Orchard => {
                self.apply_orchard_signature(signature.action_index(), spend_auth_sig)
            }
            orchard::ValuePool::Ironwood => {
                self.apply_ironwood_signature(signature.action_index(), spend_auth_sig)
            }
        }
    }

    fn generate_or_apply_orchard_signature<F>(&mut self, index: usize, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut orchard::pczt::Action, [u8; 32]) -> Result<(), orchard::pczt::SignerError>,
    {
        let action = self
            .orchard
            .bundle
            .actions_mut()
            .get_mut(index)
            .ok_or(Error::InvalidIndex)?;

        // Check consistency of the input being signed if we have its note components.
        match action.spend().verify_nullifier(None) {
            Err(
                orchard::pczt::VerifyError::MissingRecipient
                | orchard::pczt::VerifyError::MissingValue
                | orchard::pczt::VerifyError::MissingRho
                | orchard::pczt::VerifyError::MissingRandomSeed,
            ) => Ok(()),
            r => r,
        }
        .map_err(Error::OrchardVerify)?;

        // Generate or apply the signature.
        f(action, self.shielded_sighash).map_err(Error::OrchardSign)?;

        // Update transaction modifiability: all transaction effects have been committed
        // to by the signature.
        self.global.tx_modifiable &= !(FLAG_TRANSPARENT_INPUTS_MODIFIABLE
            | FLAG_TRANSPARENT_OUTPUTS_MODIFIABLE
            | FLAG_SHIELDED_MODIFIABLE);

        Ok(())
    }

    /// Signs the Ironwood spend at the given index with the given spend authorizing key.
    ///
    /// Requires the spend's `fvk` field to be set (because the API does not take an FVK).
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    ///
    /// `rng` provides the randomness for the signature.
    pub fn sign_ironwood<R: Rng + CryptoRng>(
        &mut self,
        rng: R,
        index: usize,
        ask: &orchard::keys::SpendAuthorizingKey,
    ) -> Result<(), Error> {
        self.generate_or_apply_ironwood_signature(index, |spend, shielded_sighash| {
            spend.sign(shielded_sighash, ask, rng)
        })
    }

    /// Applies the given signature to the Ironwood spend.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn apply_ironwood_signature(
        &mut self,
        index: usize,
        signature: redpallas::Signature<redpallas::SpendAuth>,
    ) -> Result<(), Error> {
        self.generate_or_apply_ironwood_signature(index, |action, shielded_sighash| {
            action.apply_signature(shielded_sighash, signature)
        })
    }

    fn generate_or_apply_ironwood_signature<F>(&mut self, index: usize, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut orchard::pczt::Action, [u8; 32]) -> Result<(), orchard::pczt::SignerError>,
    {
        let action = self
            .ironwood
            .bundle
            .actions_mut()
            .get_mut(index)
            .ok_or(Error::InvalidIndex)?;

        // Check consistency of the input being signed if we have its note components.
        match action.spend().verify_nullifier(None) {
            Err(
                orchard::pczt::VerifyError::MissingRecipient
                | orchard::pczt::VerifyError::MissingValue
                | orchard::pczt::VerifyError::MissingRho
                | orchard::pczt::VerifyError::MissingRandomSeed,
            ) => Ok(()),
            r => r,
        }
        .map_err(Error::IronwoodVerify)?;

        // Generate or apply the signature.
        f(action, self.shielded_sighash).map_err(Error::IronwoodSign)?;

        // Update transaction modifiability: all transaction effects have been committed
        // to by the signature.
        self.global.tx_modifiable &= !(FLAG_TRANSPARENT_INPUTS_MODIFIABLE
            | FLAG_TRANSPARENT_OUTPUTS_MODIFIABLE
            | FLAG_SHIELDED_MODIFIABLE);

        Ok(())
    }

    /// Finishes the Signer role, returning the updated PCZT.
    pub fn finish(self) -> Pczt {
        let Self {
            global,
            transparent,
            sapling,
            orchard,
            ironwood,
            empty_ironwood,
            tx_data: _,
            txid_parts: _,
            shielded_sighash: _,
            transparent_sighash_policy: _,
        } = self;

        Pczt {
            global,
            transparent: crate::transparent::Bundle::serialize_from(transparent),
            sapling: sapling.reserialize(),
            orchard: orchard.reserialize(),
            ironwood: empty_ironwood.unwrap_or_else(|| ironwood.reserialize()),
        }
    }
}

/// The receivers of a parsed `user_address`, in the forms needed for comparison against
/// an output's committed recipient.
///
/// A Unified Address contains at most one receiver per typecode, so a single slot per
/// value pool suffices. The Ironwood pool shares the Orchard receiver encoding.
#[derive(Default)]
struct UserAddressReceivers {
    orchard: Option<[u8; 43]>,
    sapling: Option<[u8; 43]>,
    transparent: Option<TransparentAddress>,
}

impl TryFromAddress for UserAddressReceivers {
    type Error = Infallible;

    // Sprout addresses use the default implementation: no PCZT output can commit to a
    // Sprout recipient, so conversion failure correctly reports a mismatch.

    fn try_from_sapling(
        _net: NetworkType,
        data: [u8; 43],
    ) -> Result<Self, ConversionError<Self::Error>> {
        Ok(Self {
            sapling: Some(data),
            ..Self::default()
        })
    }

    fn try_from_unified(
        _net: NetworkType,
        data: unified::Address,
    ) -> Result<Self, ConversionError<Self::Error>> {
        let mut receivers = Self::default();
        for item in data.items() {
            match item {
                Receiver::Orchard(data) => receivers.orchard = Some(data),
                Receiver::Sapling(data) => receivers.sapling = Some(data),
                Receiver::P2pkh(data) => {
                    receivers.transparent = Some(TransparentAddress::PublicKeyHash(data));
                }
                Receiver::P2sh(data) => {
                    receivers.transparent = Some(TransparentAddress::ScriptHash(data));
                }
                Receiver::Unknown { .. } => (),
            }
        }
        Ok(receivers)
    }

    fn try_from_transparent_p2pkh(
        _net: NetworkType,
        data: [u8; 20],
    ) -> Result<Self, ConversionError<Self::Error>> {
        Ok(Self {
            transparent: Some(TransparentAddress::PublicKeyHash(data)),
            ..Self::default()
        })
    }

    fn try_from_transparent_p2sh(
        _net: NetworkType,
        data: [u8; 20],
    ) -> Result<Self, ConversionError<Self::Error>> {
        Ok(Self {
            transparent: Some(TransparentAddress::ScriptHash(data)),
            ..Self::default()
        })
    }

    fn try_from_tex(
        _net: NetworkType,
        data: [u8; 20],
    ) -> Result<Self, ConversionError<Self::Error>> {
        Ok(Self {
            transparent: Some(TransparentAddress::PublicKeyHash(data)),
            ..Self::default()
        })
    }
}

/// Parses a `user_address` into its comparable receivers.
///
/// The network encoded in the address is not checked: a PCZT does not commit to a
/// network, so only receiver data can be compared.
fn parse_user_address(addr: &str) -> Result<UserAddressReceivers, UserAddressError> {
    let addr = ZcashAddress::try_from_encoded(addr).map_err(UserAddressError::Parse)?;
    // The only possible conversion failure is an address kind that cannot contain any
    // recipient representable in a PCZT output, which is a mismatch by definition.
    addr.convert()
        .map_err(|_| UserAddressError::MismatchedRecipient)
}

/// Verifies a shielded output's `user_address` against its committed recipient.
///
/// `receiver` selects, from the parsed address, the receiver encoding for the output's
/// value pool.
fn check_shielded_user_address(
    user_address: Option<&str>,
    recipient: Option<&[u8; 43]>,
    receiver: fn(&UserAddressReceivers) -> Option<[u8; 43]>,
) -> Result<(), UserAddressError> {
    match user_address {
        None => Ok(()),
        Some(addr) => {
            // A display claim that cannot be checked against the committed recipient
            // must not reach a signature.
            let recipient = recipient.ok_or(UserAddressError::MissingRecipient)?;
            if receiver(&parse_user_address(addr)?) == Some(*recipient) {
                Ok(())
            } else {
                Err(UserAddressError::MismatchedRecipient)
            }
        }
    }
}

/// Verifies a transparent output's `user_address` against its `script_pubkey`.
///
/// The output pays the claimed address only if `script_pubkey` is exactly the canonical
/// P2PKH or P2SH script for a transparent receiver of the parsed address; any other
/// script (including a non-standard one) is a mismatch.
fn check_transparent_user_address(
    user_address: Option<&str>,
    script_pubkey: &[u8],
) -> Result<(), UserAddressError> {
    match user_address {
        None => Ok(()),
        Some(addr) => match parse_user_address(addr)?.transparent {
            Some(receiver) if receiver.script().to_bytes() == script_pubkey => Ok(()),
            _ => Err(UserAddressError::MismatchedRecipient),
        },
    }
}

/// Verifies every output's `user_address` against that output's committed recipient,
/// rejecting the PCZT if any claim fails to verify.
fn verify_user_addresses(pczt: &Pczt) -> Result<(), Error> {
    for (index, output) in pczt.transparent.outputs.iter().enumerate() {
        check_transparent_user_address(output.user_address.as_deref(), &output.script_pubkey)
            .map_err(|error| Error::UserAddress {
                pool: PoolType::Transparent,
                index,
                error,
            })?;
    }

    for (index, output) in pczt.sapling.outputs.iter().enumerate() {
        check_shielded_user_address(
            output.user_address.as_deref(),
            output.recipient.as_ref(),
            |receivers| receivers.sapling,
        )
        .map_err(|error| Error::UserAddress {
            pool: PoolType::Shielded(ShieldedPool::Sapling),
            index,
            error,
        })?;
    }

    for (pool, bundle) in [
        (ShieldedPool::Orchard, &pczt.orchard),
        (ShieldedPool::Ironwood, &pczt.ironwood),
    ] {
        for (index, action) in bundle.actions.iter().enumerate() {
            check_shielded_user_address(
                action.output.user_address.as_deref(),
                action.output.recipient.as_ref(),
                |receivers| receivers.orchard,
            )
            .map_err(|error| Error::UserAddress {
                pool: PoolType::Shielded(pool),
                index,
                error,
            })?;
        }
    }

    Ok(())
}

/// Errors that can occur while creating signatures for a PCZT.
#[derive(Debug)]
pub enum Error {
    Extract(crate::ExtractError),
    InvalidIndex,
    IronwoodSign(orchard::pczt::SignerError),
    IronwoodVerify(orchard::pczt::VerifyError),
    OrchardSign(orchard::pczt::SignerError),
    OrchardVerify(orchard::pczt::VerifyError),
    SaplingSign(sapling::pczt::SignerError),
    SaplingVerify(sapling::pczt::VerifyError),
    TransparentSign(transparent::pczt::SignerError),
    /// An output's `user_address` failed verification against the output's committed
    /// recipient.
    UserAddress {
        /// The value pool of the output whose `user_address` failed verification.
        pool: PoolType,
        /// The index of the output within its bundle (for Orchard-protocol pools, the
        /// index of its action).
        index: usize,
        /// The reason verification failed.
        error: UserAddressError,
    },
}

impl From<crate::ExtractError> for Error {
    fn from(e: crate::ExtractError) -> Self {
        Error::Extract(e)
    }
}

/// Errors in verifying an output's `user_address` against the output's committed
/// recipient.
///
/// An output's `user_address` is an uncommitted display string whose contract is that it
/// contains the output's committed recipient (directly, or as a receiver within a
/// Unified Address). Each of these errors means that contract could not be confirmed, so
/// the output must not be signed.
#[derive(Debug)]
#[non_exhaustive]
pub enum UserAddressError {
    /// The `user_address` is not a valid Zcash address encoding.
    Parse(zcash_address::ParseError),
    /// The output carries a `user_address`, but the committed recipient needed to verify
    /// it is absent, so the claim cannot be checked.
    MissingRecipient,
    /// The parsed address does not contain the output's committed recipient.
    MismatchedRecipient,
}

#[cfg(all(test, feature = "io-finalizer", feature = "orchard"))]
mod tests {
    use alloc::{collections::BTreeMap, string::String};

    use ::transparent::address::TransparentAddress;
    use ff::{Field, PrimeField};
    use pasta_curves::pallas;
    use rand::{rand_core::UnwrapErr, rngs::SysRng};
    use zcash_address::{
        ToAddress, ZcashAddress,
        unified::{self, Encoding, Receiver, Uitem},
    };
    use zcash_protocol::{
        PoolType, ShieldedPool,
        consensus::{BranchId, NetworkType},
    };
    use zcash_script::script::Evaluable;

    use super::{Error, Signer, UserAddressError, check_shielded_user_address};
    use crate::{
        orchard::{Action, Spend},
        roles::{creator::Creator, io_finalizer::IoFinalizer, updater::Updater},
    };

    /// Builds a fully-dummy (zero-valued, freshly-random-key) Ironwood action, whose
    /// spend the IO Finalizer signs itself via `dummy_sk`. This lets the IO Finalizer
    /// and Signer roles be exercised without going through the transaction builder
    /// (which requires a concrete anchor to add any real spend).
    fn dummy_action() -> Action {
        let sk = orchard::keys::SpendingKey::from_bytes([7; 32]).unwrap();
        let alpha = pallas::Scalar::ONE;
        let base = crate::orchard::testing::dummy_action();

        Action {
            spend: Spend {
                alpha: Some(alpha.to_repr()),
                dummy_sk: Some(crate::common::SecretKeyBytes::new(*sk.to_bytes())),
                ..base.spend
            },
            rcv: Some([3; 32]),
            ..base
        }
    }

    /// [ZIP 374] "Anchors and pre-authorization" requires that, for a v6 transaction,
    /// signing (including the IO Finalizer's dummy-spend signing) works even while a
    /// shielded bundle's anchor is absent. The Ironwood bundle here is built by
    /// hand (rather than via the transaction builder, which requires a concrete
    /// anchor up front to add any spend) specifically to exercise that state; the
    /// full sign-then-prove-then-extract pipeline is covered by the end-to-end
    /// Ironwood tests, whose transaction builder requires the anchor to be set at the
    /// point the spend is added.
    ///
    /// [ZIP 374]: https://zips.z.cash/zip-0374#anchorsandpre-authorization
    #[test]
    fn io_finalizer_and_signer_succeed_with_absent_ironwood_anchor() {
        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(dummy_action());

        assert!(pczt.ironwood.anchor.is_none());

        let pczt = IoFinalizer::new(pczt)
            .finalize_io(UnwrapErr(SysRng))
            .unwrap();
        assert!(pczt.ironwood.anchor.is_none());
        assert!(pczt.ironwood.bsk.is_some());
        // The IO Finalizer signs and clears the dummy spending key.
        assert!(pczt.ironwood.actions[0].spend.dummy_sk.is_none());
        assert!(pczt.ironwood.actions[0].spend.spend_auth_sig.is_some());

        // `Signer::new` computes the shielded sighash and extracts transaction
        // effects; this must succeed with the anchor still absent (a v6 sighash
        // does not commit to it).
        let signer = Signer::new(pczt).unwrap();
        let pczt = signer.finish();
        assert!(pczt.ironwood.anchor.is_none());

        // The anchor may now be set by an Updater, at any point before the Prover
        // runs; nothing about signing required it to be present.
        let anchor = orchard::Anchor::empty_tree();
        let pczt = Updater::new(pczt)
            .set_ironwood_anchor(anchor)
            .unwrap()
            .finish();
        assert_eq!(pczt.ironwood.anchor, Some(anchor.to_bytes()));
    }

    /// [ZIP 326] requires the zero-valued output fabricated for a real Orchard-pool
    /// spend to carry a randomized `enc_ciphertext` instead of an encryption to the
    /// spent note's receiver. Such an output is well-formed: the Signer role neither
    /// decrypts it nor reconstructs an expected ciphertext to compare it against, so
    /// signing succeeds and the ciphertext reaches the finished PCZT byte-identical.
    ///
    /// [ZIP 326]: https://zips.z.cash/zip-0326#fabricatedsame-addressoutputsandrandomizednoteciphertexts
    #[test]
    fn signer_tolerates_fabricated_output_with_randomized_ciphertext() {
        use orchard::{
            keys::{FullViewingKey, Scope},
            note::{ExtractedNoteCommitment, NoteVersion, RandomSeed, Rho},
            note_encryption::OrchardNoteEncryption,
            value::NoteValue,
        };

        let mut action = dummy_action();

        // The fabricated output is addressed to a real receiver and carries explicit
        // `recipient`, `value` and `rseed` fields, so its note commitment is ordinary.
        let rho = Option::from(Rho::from_bytes(&action.spend.nullifier)).unwrap();
        let rseed = (0u8..)
            .find_map(|i| {
                let mut bytes = [0; 32];
                bytes[0] = i;
                Option::from(RandomSeed::from_bytes(bytes, &rho))
            })
            .unwrap();
        let recipient =
            FullViewingKey::from(&orchard::keys::SpendingKey::from_bytes([0; 32]).unwrap())
                .address_at(0u32, Scope::External);
        let value = NoteValue::from_raw(0);
        let note = Option::from(orchard::Note::from_parts(
            recipient,
            value,
            rho,
            rseed,
            NoteVersion::V3,
        ))
        .unwrap();

        // Only the ephemeral key comes from the encryptor; the ciphertext is replaced
        // with bytes that trial-decryption cannot recover a note from.
        let encryptor = OrchardNoteEncryption::new(None, note, [0; 512]);
        let mut randomized_ciphertext = encryptor.encrypt_note_plaintext().0.to_vec();
        randomized_ciphertext.fill(0xab);

        action.output.cmx = Some(ExtractedNoteCommitment::from(note.commitment()).to_bytes());
        action.output.ephemeral_key =
            <orchard::note_encryption::IronwoodDomain as zcash_note_encryption::Domain>::epk_bytes(
                encryptor.epk(),
            )
            .0;
        action.output.recipient = Some(recipient.to_raw_address_bytes());
        action.output.value = Some(value.inner());
        action.output.rseed = Some(*note.rseed().as_bytes());
        action.output.user_address = None;
        action.output.enc_ciphertext =
            crate::orchard::EncCiphertext::Encrypted(randomized_ciphertext.clone());

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(action);

        let pczt = IoFinalizer::new(pczt)
            .finalize_io(UnwrapErr(SysRng))
            .unwrap();
        let pczt = Signer::new(pczt).unwrap().finish();

        assert_eq!(
            pczt.ironwood.actions[0].output.enc_ciphertext,
            crate::orchard::EncCiphertext::Encrypted(randomized_ciphertext)
        );
        assert_eq!(pczt.ironwood.actions[0].output.value, Some(0));
    }

    /// Returns the raw encoding of the Orchard payment address derived from the given
    /// spending key bytes.
    fn orchard_recipient(sk_bytes: [u8; 32]) -> [u8; 43] {
        use orchard::keys::{FullViewingKey, Scope, SpendingKey};

        FullViewingKey::from(&SpendingKey::from_bytes(sk_bytes).unwrap())
            .address_at(0u32, Scope::External)
            .to_raw_address_bytes()
    }

    /// Encodes a mainnet Unified Address containing exactly the given Orchard receiver.
    fn unified_address_for_orchard(receiver: [u8; 43]) -> String {
        unified::Address::try_from_items(
            unified::Revision::R0,
            alloc::vec![Uitem::Data(Receiver::Orchard(receiver))],
        )
        .expect("a single Orchard receiver is a valid UA")
        .encode(&NetworkType::Main)
    }

    /// An output whose `user_address` names a different recipient than the one the
    /// sighash commits to is the approval-time deception the Signer exists to prevent;
    /// the Signer role must refuse to instantiate for such a PCZT.
    #[test]
    fn signer_rejects_user_address_omitting_committed_recipient() {
        let committed = orchard_recipient([0; 32]);
        let displayed = orchard_recipient([1; 32]);
        assert_ne!(committed, displayed);

        let mut action = dummy_action();
        action.output.recipient = Some(committed);
        action.output.user_address = Some(unified_address_for_orchard(displayed));

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(action);

        assert!(matches!(
            Signer::new(pczt),
            Err(Error::UserAddress {
                pool: PoolType::Shielded(ShieldedPool::Ironwood),
                index: 0,
                error: UserAddressError::MismatchedRecipient,
            })
        ));
    }

    /// A `user_address` on an output whose `recipient` is absent is an unverifiable
    /// display claim, and omitting `recipient` must not become a way to smuggle a
    /// deceptive string past the mismatch check.
    #[test]
    fn signer_rejects_user_address_without_committed_recipient() {
        let mut action = dummy_action();
        action.output.recipient = None;
        action.output.user_address = Some(unified_address_for_orchard(orchard_recipient([0; 32])));

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(action);

        assert!(matches!(
            Signer::new(pczt),
            Err(Error::UserAddress {
                pool: PoolType::Shielded(ShieldedPool::Ironwood),
                index: 0,
                error: UserAddressError::MissingRecipient,
            })
        ));
    }

    /// A `user_address` that is not a Zcash address encoding cannot make any verifiable
    /// claim at all.
    #[test]
    fn signer_rejects_unparseable_user_address() {
        let mut action = dummy_action();
        action.output.recipient = Some(orchard_recipient([0; 32]));
        action.output.user_address = Some("not a zcash address".into());

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(action);

        assert!(matches!(
            Signer::new(pczt),
            Err(Error::UserAddress {
                pool: PoolType::Shielded(ShieldedPool::Ironwood),
                index: 0,
                error: UserAddressError::Parse(_),
            })
        ));
    }

    /// A `user_address` containing the committed recipient satisfies the field's
    /// contract, and signing proceeds.
    #[test]
    fn signer_accepts_user_address_containing_recipient() {
        let recipient = orchard_recipient([0; 32]);

        let mut action = dummy_action();
        action.output.recipient = Some(recipient);
        action.output.user_address = Some(unified_address_for_orchard(recipient));

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.ironwood.actions.push(action);

        let pczt = IoFinalizer::new(pczt)
            .finalize_io(UnwrapErr(SysRng))
            .unwrap();
        let pczt = Signer::new(pczt).unwrap().finish();
        assert_eq!(pczt.ironwood.actions[0].output.recipient, Some(recipient));
    }

    /// The check compares the committed recipient against the receiver for the output's
    /// own value pool: a Unified Address that merely contains the committed bytes under
    /// a different pool's typecode does not contain the recipient.
    #[test]
    fn shielded_user_address_check_selects_the_pools_receiver() {
        let orchard_receiver = [1; 43];
        let sapling_receiver = [2; 43];
        let ua = unified::Address::try_from_items(
            unified::Revision::R0,
            alloc::vec![
                Uitem::Data(Receiver::Orchard(orchard_receiver)),
                Uitem::Data(Receiver::Sapling(sapling_receiver)),
            ],
        )
        .unwrap()
        .encode(&NetworkType::Main);

        assert!(
            check_shielded_user_address(Some(&ua), Some(&orchard_receiver), |r| r.orchard).is_ok()
        );
        assert!(
            check_shielded_user_address(Some(&ua), Some(&sapling_receiver), |r| r.sapling).is_ok()
        );
        assert!(matches!(
            check_shielded_user_address(Some(&ua), Some(&sapling_receiver), |r| r.orchard),
            Err(UserAddressError::MismatchedRecipient)
        ));
        assert!(matches!(
            check_shielded_user_address(Some(&ua), Some(&orchard_receiver), |r| r.sapling),
            Err(UserAddressError::MismatchedRecipient)
        ));
    }

    /// A transparent output pays the claimed address only if its `script_pubkey` is the
    /// canonical script for one of that address's transparent receivers.
    #[test]
    fn signer_rejects_mismatched_transparent_user_address() {
        let output = transparent_output(
            TransparentAddress::PublicKeyHash([1; 20]),
            ZcashAddress::from_transparent_p2pkh(NetworkType::Main, [2; 20]).encode(),
        );

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.transparent.outputs.push(output);

        assert!(matches!(
            Signer::new(pczt),
            Err(Error::UserAddress {
                pool: PoolType::Transparent,
                index: 0,
                error: UserAddressError::MismatchedRecipient,
            })
        ));
    }

    /// A transparent `user_address` whose receiver script matches `script_pubkey`
    /// satisfies the field's contract, and signing proceeds.
    #[test]
    fn signer_accepts_matching_transparent_user_address() {
        let output = transparent_output(
            TransparentAddress::PublicKeyHash([1; 20]),
            ZcashAddress::from_transparent_p2pkh(NetworkType::Main, [1; 20]).encode(),
        );

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.transparent.outputs.push(output);

        let _ = Signer::new(pczt).unwrap().finish();
    }

    /// A Sapling output's `user_address` is checked against the Sapling recipient, and a
    /// failure is attributed to the Sapling pool.
    #[test]
    fn signer_rejects_mismatched_sapling_user_address() {
        let output = crate::sapling::Output {
            cv: [0; 32],
            cmu: [1; 32],
            ephemeral_key: [2; 32],
            enc_ciphertext: alloc::vec![3; 580],
            out_ciphertext: alloc::vec![4; 80],
            zkproof: None,
            recipient: Some([9; 43]),
            value: None,
            rseed: None,
            rcv: None,
            ock: None,
            zip32_derivation: None,
            user_address: Some(ZcashAddress::from_sapling(NetworkType::Main, [8; 43]).encode()),
            proprietary: BTreeMap::new(),
        };

        let mut pczt = Creator::new(BranchId::Nu6_3.into(), 100, 133, None, None)
            .unwrap()
            .build()
            .unwrap();
        pczt.sapling.outputs.push(output);

        assert!(matches!(
            Signer::new(pczt),
            Err(Error::UserAddress {
                pool: PoolType::Shielded(ShieldedPool::Sapling),
                index: 0,
                error: UserAddressError::MismatchedRecipient,
            })
        ));
    }

    /// Builds a transparent output paying `addr`, with the given display string.
    fn transparent_output(
        addr: TransparentAddress,
        user_address: String,
    ) -> crate::transparent::Output {
        crate::transparent::Output {
            value: 100,
            script_pubkey: addr.script().to_bytes(),
            redeem_script: None,
            bip32_derivation: BTreeMap::new(),
            user_address: Some(user_address),
            proprietary: BTreeMap::new(),
        }
    }
}
