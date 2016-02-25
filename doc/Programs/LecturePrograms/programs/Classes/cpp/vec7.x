	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 11
	.section	__TEXT,__literal16,16byte_literals
	.align	4
LCPI0_0:
	.long	1127219200              ## 0x43300000
	.long	1160773632              ## 0x45300000
	.long	0                       ## 0x0
	.long	0                       ## 0x0
LCPI0_1:
	.quad	4841369599423283200     ## double 4.503600e+15
	.quad	4985484787499139072     ## double 1.934281e+25
	.section	__TEXT,__literal8,8byte_literals
	.align	3
LCPI0_2:
	.quad	4696837146684686336     ## double 1.0E+6
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	4, 0x90
_main:                                  ## @main
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0
## BB#0:                                ## %.preheader
	pushq	%rbp
Ltmp8:
	.cfi_def_cfa_offset 16
Ltmp9:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp10:
	.cfi_def_cfa_register %rbp
	pushq	%r14
	pushq	%rbx
	subq	$32, %rsp
Ltmp11:
	.cfi_offset %rbx, -32
Ltmp12:
	.cfi_offset %r14, -24
	callq	_clock
	movq	%rax, %rbx
	callq	_clock
	subq	%rbx, %rax
	movd	%rax, %xmm0
	punpckldq	LCPI0_0(%rip), %xmm0 ## xmm0 = xmm0[0],mem[0],xmm0[1],mem[1]
	subpd	LCPI0_1(%rip), %xmm0
	haddpd	%xmm0, %xmm0
	divsd	LCPI0_2(%rip), %xmm0
	movaps	%xmm0, -48(%rbp)        ## 16-byte Spill
	movq	__ZNSt3__14coutE@GOTPCREL(%rip), %rdi
	movq	(%rdi), %rax
	movq	-24(%rax), %rax
	orl	$17408, 8(%rax,%rdi)    ## imm = 0x4400
	movq	(%rdi), %rax
	movq	-24(%rax), %rax
	movq	$10, 16(%rax,%rdi)
	movq	(%rdi), %rax
	movq	-24(%rax), %rax
	movq	$20, 24(%rax,%rdi)
	leaq	L_.str(%rip), %rsi
	movl	$31, %edx
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movaps	-48(%rbp), %xmm0        ## 16-byte Reload
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEd
	movq	%rax, %rbx
	movq	(%rbx), %rax
	movq	-24(%rax), %rsi
	addq	%rbx, %rsi
	leaq	-24(%rbp), %r14
	movq	%r14, %rdi
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp0:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	movq	%r14, %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp1:
## BB#1:
	movq	(%rax), %rcx
	movq	56(%rcx), %rcx
Ltmp2:
	movl	$10, %esi
	movq	%rax, %rdi
	callq	*%rcx
	movb	%al, %r14b
Ltmp3:
## BB#2:                                ## %_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc.exit
	leaq	-24(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
	movsbl	%r14b, %esi
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc
	movq	%rbx, %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv
	xorl	%eax, %eax
	addq	$32, %rsp
	popq	%rbx
	popq	%r14
	popq	%rbp
	retq
LBB0_3:
Ltmp4:
	movq	%rax, %rbx
Ltmp5:
	leaq	-24(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp6:
## BB#4:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB0_5:
Ltmp7:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table0:
Lexception0:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.byte	73                      ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.byte	65                      ## Call site table length
Lset0 = Lfunc_begin0-Lfunc_begin0       ## >> Call Site 1 <<
	.long	Lset0
Lset1 = Ltmp0-Lfunc_begin0              ##   Call between Lfunc_begin0 and Ltmp0
	.long	Lset1
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset2 = Ltmp0-Lfunc_begin0              ## >> Call Site 2 <<
	.long	Lset2
Lset3 = Ltmp3-Ltmp0                     ##   Call between Ltmp0 and Ltmp3
	.long	Lset3
Lset4 = Ltmp4-Lfunc_begin0              ##     jumps to Ltmp4
	.long	Lset4
	.byte	0                       ##   On action: cleanup
Lset5 = Ltmp3-Lfunc_begin0              ## >> Call Site 3 <<
	.long	Lset5
Lset6 = Ltmp5-Ltmp3                     ##   Call between Ltmp3 and Ltmp5
	.long	Lset6
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset7 = Ltmp5-Lfunc_begin0              ## >> Call Site 4 <<
	.long	Lset7
Lset8 = Ltmp6-Ltmp5                     ##   Call between Ltmp5 and Ltmp6
	.long	Lset8
Lset9 = Ltmp7-Lfunc_begin0              ##     jumps to Ltmp7
	.long	Lset9
	.byte	1                       ##   On action: 1
Lset10 = Ltmp6-Lfunc_begin0             ## >> Call Site 5 <<
	.long	Lset10
Lset11 = Lfunc_end0-Ltmp6               ##   Call between Ltmp6 and Lfunc_end0
	.long	Lset11
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.weak_def_can_be_hidden	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.align	4, 0x90
__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m: ## @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception1
## BB#0:
	pushq	%rbp
Ltmp43:
	.cfi_def_cfa_offset 16
Ltmp44:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp45:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
Ltmp46:
	.cfi_offset %rbx, -56
Ltmp47:
	.cfi_offset %r12, -48
Ltmp48:
	.cfi_offset %r13, -40
Ltmp49:
	.cfi_offset %r14, -32
Ltmp50:
	.cfi_offset %r15, -24
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
Ltmp13:
	leaq	-64(%rbp), %rdi
	movq	%rbx, %rsi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_
Ltmp14:
## BB#1:
	cmpb	$0, -64(%rbp)
	je	LBB1_13
## BB#2:
	movq	(%rbx), %rax
	movq	-24(%rax), %r12
	movq	40(%r12,%rbx), %rdi
	movl	$176, %eax
	andl	8(%r12,%rbx), %eax
	cmpl	$32, %eax
	movq	%r15, %r13
	jne	LBB1_4
## BB#3:
	leaq	(%r15,%r14), %r13
LBB1_4:
	leaq	(%rbx,%r12), %r8
	movl	144(%rbx,%r12), %eax
	cmpl	$-1, %eax
	jne	LBB1_10
## BB#5:
Ltmp15:
	movq	%rdi, -72(%rbp)         ## 8-byte Spill
	leaq	-48(%rbp), %rdi
	movq	%r8, %rsi
	movq	%r8, -80(%rbp)          ## 8-byte Spill
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp16:
## BB#6:                                ## %.noexc
Ltmp17:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	leaq	-48(%rbp), %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp18:
## BB#7:
	movq	(%rax), %rcx
	movq	56(%rcx), %rcx
Ltmp19:
	movl	$32, %esi
	movq	%rax, %rdi
	callq	*%rcx
	movb	%al, -81(%rbp)          ## 1-byte Spill
Ltmp20:
## BB#8:                                ## %_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenEc.exit.i
Ltmp25:
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp26:
## BB#9:                                ## %.noexc1
	movsbl	-81(%rbp), %eax         ## 1-byte Folded Reload
	movl	%eax, 144(%rbx,%r12)
	movq	-72(%rbp), %rdi         ## 8-byte Reload
	movq	-80(%rbp), %r8          ## 8-byte Reload
LBB1_10:
	addq	%r15, %r14
Ltmp27:
	movsbl	%al, %r9d
	movq	%r15, %rsi
	movq	%r13, %rdx
	movq	%r14, %rcx
	callq	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Ltmp28:
## BB#11:
	testq	%rax, %rax
	jne	LBB1_13
## BB#12:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %rdi
	movl	32(%rbx,%rax), %esi
	orl	$5, %esi
Ltmp29:
	callq	__ZNSt3__18ios_base5clearEj
Ltmp30:
LBB1_13:                                ## %_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateEj.exit
Ltmp34:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
Ltmp35:
LBB1_21:
	movq	%rbx, %rax
	addq	$56, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB1_18:
Ltmp36:
	movq	%rax, %r14
	jmp	LBB1_19
LBB1_16:
Ltmp31:
	movq	%rax, %r14
	jmp	LBB1_17
LBB1_14:
Ltmp21:
	movq	%rax, %r14
Ltmp22:
	leaq	-48(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
Ltmp23:
LBB1_17:                                ## %.body
Ltmp32:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
Ltmp33:
LBB1_19:
	movq	%r14, %rdi
	callq	___cxa_begin_catch
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	addq	-24(%rax), %rdi
Ltmp37:
	callq	__ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv
Ltmp38:
## BB#20:
	callq	___cxa_end_catch
	jmp	LBB1_21
LBB1_22:
Ltmp39:
	movq	%rax, %rbx
Ltmp40:
	callq	___cxa_end_catch
Ltmp41:
## BB#23:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB1_24:
Ltmp42:
	movq	%rax, %rdi
	callq	___clang_call_terminate
LBB1_15:
Ltmp24:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end1:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table1:
Lexception1:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.asciz	"\245\201\200\200"      ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.ascii	"\234\001"              ## Call site table length
Lset12 = Ltmp13-Lfunc_begin1            ## >> Call Site 1 <<
	.long	Lset12
Lset13 = Ltmp14-Ltmp13                  ##   Call between Ltmp13 and Ltmp14
	.long	Lset13
Lset14 = Ltmp36-Lfunc_begin1            ##     jumps to Ltmp36
	.long	Lset14
	.byte	1                       ##   On action: 1
Lset15 = Ltmp15-Lfunc_begin1            ## >> Call Site 2 <<
	.long	Lset15
Lset16 = Ltmp16-Ltmp15                  ##   Call between Ltmp15 and Ltmp16
	.long	Lset16
Lset17 = Ltmp31-Lfunc_begin1            ##     jumps to Ltmp31
	.long	Lset17
	.byte	1                       ##   On action: 1
Lset18 = Ltmp17-Lfunc_begin1            ## >> Call Site 3 <<
	.long	Lset18
Lset19 = Ltmp20-Ltmp17                  ##   Call between Ltmp17 and Ltmp20
	.long	Lset19
Lset20 = Ltmp21-Lfunc_begin1            ##     jumps to Ltmp21
	.long	Lset20
	.byte	1                       ##   On action: 1
Lset21 = Ltmp25-Lfunc_begin1            ## >> Call Site 4 <<
	.long	Lset21
Lset22 = Ltmp30-Ltmp25                  ##   Call between Ltmp25 and Ltmp30
	.long	Lset22
Lset23 = Ltmp31-Lfunc_begin1            ##     jumps to Ltmp31
	.long	Lset23
	.byte	1                       ##   On action: 1
Lset24 = Ltmp34-Lfunc_begin1            ## >> Call Site 5 <<
	.long	Lset24
Lset25 = Ltmp35-Ltmp34                  ##   Call between Ltmp34 and Ltmp35
	.long	Lset25
Lset26 = Ltmp36-Lfunc_begin1            ##     jumps to Ltmp36
	.long	Lset26
	.byte	1                       ##   On action: 1
Lset27 = Ltmp22-Lfunc_begin1            ## >> Call Site 6 <<
	.long	Lset27
Lset28 = Ltmp23-Ltmp22                  ##   Call between Ltmp22 and Ltmp23
	.long	Lset28
Lset29 = Ltmp24-Lfunc_begin1            ##     jumps to Ltmp24
	.long	Lset29
	.byte	1                       ##   On action: 1
Lset30 = Ltmp32-Lfunc_begin1            ## >> Call Site 7 <<
	.long	Lset30
Lset31 = Ltmp33-Ltmp32                  ##   Call between Ltmp32 and Ltmp33
	.long	Lset31
Lset32 = Ltmp42-Lfunc_begin1            ##     jumps to Ltmp42
	.long	Lset32
	.byte	1                       ##   On action: 1
Lset33 = Ltmp33-Lfunc_begin1            ## >> Call Site 8 <<
	.long	Lset33
Lset34 = Ltmp37-Ltmp33                  ##   Call between Ltmp33 and Ltmp37
	.long	Lset34
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset35 = Ltmp37-Lfunc_begin1            ## >> Call Site 9 <<
	.long	Lset35
Lset36 = Ltmp38-Ltmp37                  ##   Call between Ltmp37 and Ltmp38
	.long	Lset36
Lset37 = Ltmp39-Lfunc_begin1            ##     jumps to Ltmp39
	.long	Lset37
	.byte	0                       ##   On action: cleanup
Lset38 = Ltmp38-Lfunc_begin1            ## >> Call Site 10 <<
	.long	Lset38
Lset39 = Ltmp40-Ltmp38                  ##   Call between Ltmp38 and Ltmp40
	.long	Lset39
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset40 = Ltmp40-Lfunc_begin1            ## >> Call Site 11 <<
	.long	Lset40
Lset41 = Ltmp41-Ltmp40                  ##   Call between Ltmp40 and Ltmp41
	.long	Lset41
Lset42 = Ltmp42-Lfunc_begin1            ##     jumps to Ltmp42
	.long	Lset42
	.byte	1                       ##   On action: 1
Lset43 = Ltmp41-Lfunc_begin1            ## >> Call Site 12 <<
	.long	Lset43
Lset44 = Lfunc_end1-Ltmp41              ##   Call between Ltmp41 and Lfunc_end1
	.long	Lset44
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.private_extern	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.globl	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.weak_def_can_be_hidden	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.align	4, 0x90
__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_: ## @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception2
## BB#0:
	pushq	%rbp
Ltmp57:
	.cfi_def_cfa_offset 16
Ltmp58:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp59:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$40, %rsp
Ltmp60:
	.cfi_offset %rbx, -56
Ltmp61:
	.cfi_offset %r12, -48
Ltmp62:
	.cfi_offset %r13, -40
Ltmp63:
	.cfi_offset %r14, -32
Ltmp64:
	.cfi_offset %r15, -24
	movq	%r8, %r14
	movq	%rcx, %r13
	movq	%rdi, %rbx
	xorl	%eax, %eax
	testq	%rbx, %rbx
	je	LBB2_12
## BB#1:
	movq	%r13, %rax
	subq	%rsi, %rax
	movq	24(%r14), %rcx
	xorl	%r15d, %r15d
	subq	%rax, %rcx
	cmovgq	%rcx, %r15
	movq	%rdx, %r12
	subq	%rsi, %r12
	testq	%r12, %r12
	jle	LBB2_3
## BB#2:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movq	%r13, -80(%rbp)         ## 8-byte Spill
	movq	%rdx, -72(%rbp)         ## 8-byte Spill
	movq	%r12, %rdx
	movl	%r9d, %r13d
	callq	*96(%rax)
	movl	%r13d, %r9d
	movq	-72(%rbp), %rdx         ## 8-byte Reload
	movq	-80(%rbp), %r13         ## 8-byte Reload
	movq	%rax, %rcx
	xorl	%eax, %eax
	cmpq	%r12, %rcx
	jne	LBB2_12
LBB2_3:
	testq	%r15, %r15
	jle	LBB2_9
## BB#4:
	movq	%rdx, -72(%rbp)         ## 8-byte Spill
	movq	%r14, %r12
	movsbl	%r9b, %edx
	leaq	-64(%rbp), %rdi
	movq	%r15, %rsi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc
	testb	$1, -64(%rbp)
	jne	LBB2_5
## BB#6:
	leaq	-63(%rbp), %rsi
	jmp	LBB2_7
LBB2_5:
	movq	-48(%rbp), %rsi
LBB2_7:                                 ## %_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataEv.exit
	movq	(%rbx), %rax
	movq	96(%rax), %rax
Ltmp51:
	movq	%rbx, %rdi
	movq	%r15, %rdx
	callq	*%rax
	movq	%rax, %r14
Ltmp52:
## BB#8:                                ## %_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnEPKcl.exit
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev
	xorl	%eax, %eax
	cmpq	%r15, %r14
	movq	%r12, %r14
	movq	-72(%rbp), %rdx         ## 8-byte Reload
	jne	LBB2_12
LBB2_9:                                 ## %.thread
	subq	%rdx, %r13
	testq	%r13, %r13
	jle	LBB2_11
## BB#10:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movq	%rdx, %rsi
	movq	%r13, %rdx
	callq	*96(%rax)
	movq	%rax, %rcx
	xorl	%eax, %eax
	cmpq	%r13, %rcx
	jne	LBB2_12
LBB2_11:
	movq	$0, 24(%r14)
	movq	%rbx, %rax
LBB2_12:
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB2_13:
Ltmp53:
	movq	%rax, %rbx
Ltmp54:
	leaq	-64(%rbp), %rdi
	callq	__ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev
Ltmp55:
## BB#14:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
LBB2_15:
Ltmp56:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end2:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.align	2
GCC_except_table2:
Lexception2:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.byte	73                      ## @TType base offset
	.byte	3                       ## Call site Encoding = udata4
	.byte	65                      ## Call site table length
Lset45 = Lfunc_begin2-Lfunc_begin2      ## >> Call Site 1 <<
	.long	Lset45
Lset46 = Ltmp51-Lfunc_begin2            ##   Call between Lfunc_begin2 and Ltmp51
	.long	Lset46
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset47 = Ltmp51-Lfunc_begin2            ## >> Call Site 2 <<
	.long	Lset47
Lset48 = Ltmp52-Ltmp51                  ##   Call between Ltmp51 and Ltmp52
	.long	Lset48
Lset49 = Ltmp53-Lfunc_begin2            ##     jumps to Ltmp53
	.long	Lset49
	.byte	0                       ##   On action: cleanup
Lset50 = Ltmp52-Lfunc_begin2            ## >> Call Site 3 <<
	.long	Lset50
Lset51 = Ltmp54-Ltmp52                  ##   Call between Ltmp52 and Ltmp54
	.long	Lset51
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lset52 = Ltmp54-Lfunc_begin2            ## >> Call Site 4 <<
	.long	Lset52
Lset53 = Ltmp55-Ltmp54                  ##   Call between Ltmp54 and Ltmp55
	.long	Lset53
Lset54 = Ltmp56-Lfunc_begin2            ##     jumps to Ltmp56
	.long	Lset54
	.byte	1                       ##   On action: 1
Lset55 = Ltmp55-Lfunc_begin2            ## >> Call Site 5 <<
	.long	Lset55
Lset56 = Lfunc_end2-Ltmp55              ##   Call between Ltmp55 and Lfunc_end2
	.long	Lset56
	.long	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
	.align	2

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.private_extern	___clang_call_terminate
	.globl	___clang_call_terminate
	.weak_def_can_be_hidden	___clang_call_terminate
	.align	4, 0x90
___clang_call_terminate:                ## @__clang_call_terminate
## BB#0:
	pushq	%rbp
	movq	%rsp, %rbp
	callq	___cxa_begin_catch
	callq	__ZSt9terminatev

	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"Time used  for vector addition="


.subsections_via_symbols
